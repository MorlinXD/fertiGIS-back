
# app/main.py
from fastapi import FastAPI, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import text
from fastapi.middleware.cors import CORSMiddleware
import json
import joblib
from sklearn.preprocessing import StandardScaler
import numpy as np

from pydantic import BaseModel

from app.db import get_db
from app.schemas import ParcelaUpdate, ParcelaIn
from app import crud


app = FastAPI(title="Banano GIS API", version="1.0.0")

app.add_middleware(
  CORSMiddleware,
  allow_origins=[
    "http://localhost:5173",
    "http://127.0.0.1:5173",
  ],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

# Cargar el modelo
model = joblib.load('Modelo/modelo_fertilizacion_optimizado.pkl')

# Escalador (si durante el entrenamiento se usó uno, debería cargarse aquí en vez de instanciar uno vacío)
scaler = StandardScaler()


# Definir el modelo de entrada para la API
class LoteInput(BaseModel):
  ph: float
  mo: float
  nh4: float
  p: float
  k: float
  ca: float
  mg: float
  zn: float
  cu: float
  fe: float
  mn: float
  b: float
  s: float
  racimos_ha: float
  peso_racimo: float
  has: float


# Endpoint para hacer predicciones
@app.post("/prediccion_fertilizante")
def predecir_fertilizante(input_data: LoteInput):
    input_array = np.array([[ 
        input_data.ph,
        input_data.mo,
        input_data.nh4,
        input_data.p,
        input_data.k,
        input_data.ca,
        input_data.mg,
        input_data.zn,
        input_data.cu,
        input_data.fe,
        input_data.mn,
        input_data.b,
        input_data.s,
        input_data.racimos_ha,
        input_data.peso_racimo,
        input_data.has
    ]])
    
    # Escalar los datos
    input_array_scaled = scaler.fit_transform(input_array)

    # Realizar la predicción
    prediccion = model.predict(input_array_scaled)

    # Devolver la predicción de fertilizante
    return {"prediccion_fertilizante": np.expm1(prediccion[0])}



@app.post("/parcelas", tags=["Parcelas"])
def upsert_parcela(payload: ParcelaIn, db: Session = Depends(get_db)):
  # GeoJSON -> PostGIS
  try:
    geom_json = json.dumps(payload.geom)
  except Exception:
    raise HTTPException(status_code=400, detail="geom debe ser un objeto GeoJSON válido")

  sql = text("""
  INSERT INTO parcelas (lote, nombre, geom)
  VALUES (:lote, :nombre, ST_SetSRID(ST_GeomFromGeoJSON(:geom), 4326))
  ON CONFLICT (lote) DO UPDATE SET
    nombre = COALESCE(EXCLUDED.nombre, parcelas.nombre),
    geom = EXCLUDED.geom
  RETURNING lote;
  """)

  row = db.execute(sql, {
    "lote": payload.lote.strip(),
    "nombre": payload.nombre,
    "geom": geom_json
  }).mappings().first()

  if not row:
    raise HTTPException(status_code=400, detail="No se pudo guardar la parcela")

  db.commit()
  return {"ok": True, "lote": row["lote"]}


@app.patch("/parcelas/{lote}", tags=["Parcelas"])
def editar_parcela(lote: str, payload: ParcelaUpdate, db: Session = Depends(get_db)):
  row = crud.update_parcela(db, lote.strip(), payload.nombre, payload.geom)
  if not row:
    raise HTTPException(status_code=404, detail="Lote no encontrado")
  return {
    "type": "Feature",
    "geometry": row["geom"],
    "properties": {
      "parcela_id": str(row["parcela_id"]),
      "lote": row["lote"],
      "nombre": row["nombre"],
      "has": float(row["has"]) if row["has"] is not None else None
    }
  }


@app.get("/mapa/lotes", tags=["Mapa"])
def mapa_lotes(anio: int, mes: int, variable: str = "produccion_lbs_ha", db: Session = Depends(get_db)):
  try:
    fc = crud.get_mapa_lotes_geojson(db, anio, mes, variable)
    return fc
  except ValueError as ve:
    raise HTTPException(status_code=400, detail=str(ve))
  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))


# ✅ Para el MAPA: devuelve SOLO parcelas con geometría (las que realmente se pueden dibujar)
@app.get("/parcelas", tags=["Parcelas"])
def listar_parcelas_con_geom(db: Session = Depends(get_db)):
  sql = text("""
    SELECT jsonb_build_object(
    'type','FeatureCollection',
    'features', COALESCE(jsonb_agg(
      jsonb_build_object(
      'type','Feature',
      'geometry', ST_AsGeoJSON(geom)::jsonb,
      'properties', jsonb_build_object(
        'parcela_id', parcela_id,
        'lote', lote,
        'nombre', nombre,
        'has', has
      )
      )
    ), '[]'::jsonb)
    ) AS fc
    FROM parcelas
    WHERE geom IS NOT NULL;
  """)
  row = db.execute(sql).mappings().first()
  return row["fc"]


# ✅ Por si quieres ver TODAS, aunque no tengan geometría (para debugging)
@app.get("/parcelas/todas", tags=["Parcelas"])
def listar_todas_parcelas(db: Session = Depends(get_db)):
    sql = text("""
      SELECT lote, nombre, has, (geom IS NOT NULL) AS tiene_geom
      FROM parcelas
      ORDER BY lote::int;
    """)
    rows = db.execute(sql).mappings().all()
    return {"total": len(rows), "items": rows}

@app.get("/suelos/resumen", tags=["Suelos"])
def suelos_resumen(db: Session = Depends(get_db)):
    sql = text("""
      SELECT
        p.lote,
        p.nombre,
        p.has,

        -- principales
        s.ph,
        s.mo            AS mat_org_pct,
        s.nh4           AS n_suelo_ppm,
        s.p             AS p_suelo_ppm,
        s.k             AS k_suelo_meq_100g,

        -- textura
        s.arena,
        s.limo,
        s.arcilla,

        -- micronutrientes
        s.zn            AS zn_suelo_ppm,
        s.cu            AS cu_suelo_ppm,
        s.fe            AS fe_suelo_ppm,
        s.mn            AS mn_suelo_ppm,
        s.b             AS b_suelo_ppm,
        s.s             AS s_suelo_ppm,
        s.ca            AS ca_suelo,
        s.mg            AS mg_suelo,

        -- alertas (CORRECTAMENTE ubicadas)
        (
          (CASE WHEN s.ph IS NOT NULL AND (s.ph < 5.5 OR s.ph > 7.2) THEN 1 ELSE 0 END) +
          (CASE WHEN s.mo IS NOT NULL AND s.mo < 2.0 THEN 1 ELSE 0 END) +
          (CASE WHEN s.p  IS NOT NULL AND s.p  < 10  THEN 1 ELSE 0 END) +
          (CASE WHEN s.k  IS NOT NULL AND s.k  < 0.4 THEN 1 ELSE 0 END)
        ) AS alertas_count,

        jsonb_build_object(
          'ph_fuera_rango', (s.ph IS NOT NULL AND (s.ph < 5.5 OR s.ph > 7.2)),
          'mo_baja',        (s.mo IS NOT NULL AND s.mo < 2.0),
          'p_bajo',         (s.p  IS NOT NULL AND s.p < 10),
          'k_bajo',         (s.k  IS NOT NULL AND s.k < 0.4),
          'textura_liviana',(s.arena IS NOT NULL AND s.arena > 60)
        ) AS alertas_detalle,

        -- sin fechas
        NULL::int AS anio,
        NULL::int AS mes

      FROM parcelas p
      LEFT JOIN banano_suelos s ON s.lote = p.lote
      ORDER BY p.lote::int;
    """)
    rows = db.execute(sql).mappings().all()
    return {"total": len(rows), "items": rows}

@app.get("/suelos/lote/{lote}", tags=["Suelos"])
def suelos_hist_lote(lote: str, db: Session = Depends(get_db)):
    sql = text("""
      SELECT
        p.lote,
        p.nombre,
        p.has,
        s.ph, s.mo, s.nh4, s.p, s.k, s.ca, s.mg,
        s.zn, s.cu, s.fe, s.mn, s.b, s.s,
        b.produccion_total_lbs,  -- Aquí estamos accediendo a la tabla de producción
        b.racimos_ha, b.peso_racimo
      FROM parcelas p
      LEFT JOIN banano_suelos s ON s.lote = p.lote
      LEFT JOIN banano_produccion_mensual b ON b.lote = p.lote
      WHERE p.lote = :lote
      ORDER BY b.anio DESC, b.mes DESC;
    """)
    rows = db.execute(sql, {"lote": lote.strip()}).mappings().all()

    if not rows:
        raise HTTPException(status_code=404, detail="No se encontraron datos para el lote")

    return {"lote": lote.strip(), "total": len(rows), "items": rows}


@app.get("/mapa/suelos", tags=["Mapa"])
def mapa_suelos(db: Session = Depends(get_db)):
    sql = text("""
      WITH base AS (
        SELECT
          p.lote,
          p.nombre,
          p.has,
          p.geom,
          s.ph, s.mo, s.p, s.k,
          s.arena, s.limo, s.arcilla,
          s.zn, s.cu, s.fe, s.mn, s.b, s.s, s.ca, s.mg,

          (
            (CASE WHEN s.ph IS NOT NULL AND (s.ph < 5.5 OR s.ph > 7.2) THEN 1 ELSE 0 END) +
            (CASE WHEN s.mo IS NOT NULL AND s.mo < 2.0 THEN 1 ELSE 0 END) +
            (CASE WHEN s.p  IS NOT NULL AND s.p  < 10  THEN 1 ELSE 0 END) +
            (CASE WHEN s.k  IS NOT NULL AND s.k  < 0.4 THEN 1 ELSE 0 END)
          ) AS alertas_count,

          jsonb_build_object(
            'ph_fuera_rango', (s.ph IS NOT NULL AND (s.ph < 5.5 OR s.ph > 7.2)),
            'mo_baja',        (s.mo IS NOT NULL AND s.mo < 2.0),
            'p_bajo',         (s.p  IS NOT NULL AND s.p < 10),
            'k_bajo',         (s.k  IS NOT NULL AND s.k < 0.4),
            'textura_liviana',(s.arena IS NOT NULL AND s.arena > 60)
          ) AS alertas_detalle
        FROM parcelas p
        LEFT JOIN banano_suelos s ON s.lote = p.lote
        WHERE p.geom IS NOT NULL
      )
      SELECT jsonb_build_object(
        'type','FeatureCollection',
        'features', COALESCE(jsonb_agg(
          jsonb_build_object(
            'type','Feature',
            'geometry', ST_AsGeoJSON(geom)::jsonb,
            'properties', jsonb_build_object(
              'lote', lote,
              'nombre', nombre,
              'has', has,
              'ph', ph,
              'mo', mo,
              'p', p,
              'k', k,
              'arena', arena, 'limo', limo, 'arcilla', arcilla,
              'alertas_count', alertas_count,
              'alertas_detalle', alertas_detalle
            )
          )
        ), '[]'::jsonb)
      ) AS fc
      FROM base;
    """)
    row = db.execute(sql).mappings().first()
    return row["fc"]


@app.get("/parcelas/ubicaciones", tags=["Parcelas"])
def parcelas_ubicaciones(db: Session = Depends(get_db)):
    sql = text("""
      SELECT
        lote,
        nombre,
        has,
        (geom IS NOT NULL) AS tiene_geom,
        CASE
          WHEN geom IS NOT NULL THEN ST_X(ST_Centroid(geom))
          ELSE NULL
        END AS lon,
        CASE
          WHEN geom IS NOT NULL THEN ST_Y(ST_Centroid(geom))
          ELSE NULL
        END AS lat
      FROM parcelas
      ORDER BY lote::int;
    """)
    rows = db.execute(sql).mappings().all()
    return {"total": len(rows), "items": rows}

@app.get("/mapa/suelos/var", tags=["Mapa"])
def mapa_suelos_var(
    var: str = Query("ph", description="Variable de suelo a mapear"),
    db: Session = Depends(get_db)
):
    # ✅ Lista permitida (variable -> (expresión SQL, etiqueta, unidad))
    VARS = {
        "ph":      ("s.ph", "pH", ""),
        "mo":      ("s.mo", "Materia orgánica", "%"),
        "arena":   ("s.arena", "Arena", "%"),
        "limo":    ("s.limo", "Limo", "%"),
        "arcilla": ("s.arcilla", "Arcilla", "%"),
        "nh4":     ("s.nh4", "NH4", "ppm"),
        "p":       ("s.p", "Fósforo (P)", "ppm"),
        "k":       ("s.k", "Potasio (K)", "meq/100g"),
        "ca":      ("s.ca", "Calcio (Ca)", "meq/100g"),
        "mg":      ("s.mg", "Magnesio (Mg)", "meq/100g"),
        "zn":      ("s.zn", "Zinc (Zn)", "ppm"),
        "cu":      ("s.cu", "Cobre (Cu)", "ppm"),
        "fe":      ("s.fe", "Hierro (Fe)", "ppm"),
        "mn":      ("s.mn", "Manganeso (Mn)", "ppm"),
        "b":       ("s.b", "Boro (B)", "ppm"),
        "s":       ("s.s", "Azufre (S)", "ppm"),
        "alertas": ("""
            (
              (CASE WHEN s.ph IS NOT NULL AND (s.ph < 5.5 OR s.ph > 7.2) THEN 1 ELSE 0 END) +
              (CASE WHEN s.mo IS NOT NULL AND s.mo < 2.0 THEN 1 ELSE 0 END) +
              (CASE WHEN s.p  IS NOT NULL AND s.p  < 10  THEN 1 ELSE 0 END) +
              (CASE WHEN s.k  IS NOT NULL AND s.k  < 0.4 THEN 1 ELSE 0 END)
            )
        """, "Alertas", "count"),
    }

    if var not in VARS:
        raise HTTPException(status_code=400, detail=f"Variable inválida: {var}")

    expr, label, unit = VARS[var]

    sql = text(f"""
      WITH base AS (
        SELECT
          p.lote,
          p.nombre,
          p.has,
          p.geom,

          -- valores base
          s.ph, s.mo, s.arena, s.limo, s.arcilla, s.nh4,
          s.p, s.k, s.ca, s.mg, s.zn, s.cu, s.fe, s.mn, s.b, s.s,

          -- variable seleccionada
          ({expr})::numeric AS valor,

          -- alertas (para popup)
          (
            (CASE WHEN s.ph IS NOT NULL AND (s.ph < 5.5 OR s.ph > 7.2) THEN 1 ELSE 0 END) +
            (CASE WHEN s.mo IS NOT NULL AND s.mo < 2.0 THEN 1 ELSE 0 END) +
            (CASE WHEN s.p  IS NOT NULL AND s.p  < 10  THEN 1 ELSE 0 END) +
            (CASE WHEN s.k  IS NOT NULL AND s.k  < 0.4 THEN 1 ELSE 0 END)
          ) AS alertas_count,

          jsonb_build_object(
            'ph_fuera_rango', (s.ph IS NOT NULL AND (s.ph < 5.5 OR s.ph > 7.2)),
            'mo_baja',        (s.mo IS NOT NULL AND s.mo < 2.0),
            'p_bajo',         (s.p  IS NOT NULL AND s.p < 10),
            'k_bajo',         (s.k  IS NOT NULL AND s.k < 0.4),
            'textura_liviana',(s.arena IS NOT NULL AND s.arena > 60)
          ) AS alertas_detalle

        FROM parcelas p
        LEFT JOIN banano_suelos s ON s.lote = p.lote
        WHERE p.geom IS NOT NULL
      )
      SELECT jsonb_build_object(
        'type','FeatureCollection',
        'meta', jsonb_build_object(
          'var', :var,
          'label', :label,
          'unit', :unit
        ),
        'features', COALESCE(jsonb_agg(
          jsonb_build_object(
            'type','Feature',
            'geometry', ST_AsGeoJSON(geom)::jsonb,
            'properties', jsonb_build_object(
              'lote', lote,
              'nombre', nombre,
              'has', has,

              'valor', valor,
              'var', :var,
              'label', :label,
              'unit', :unit,

              'ph', ph, 'mo', mo,
              'arena', arena, 'limo', limo, 'arcilla', arcilla,
              'nh4', nh4, 'p', p, 'k', k,
              'ca', ca, 'mg', mg,
              'zn', zn, 'cu', cu, 'fe', fe, 'mn', mn, 'b', b, 's', s,

              'alertas_count', alertas_count,
              'alertas_detalle', alertas_detalle
            )
          )
        ), '[]'::jsonb)
      ) AS fc
      FROM base;
    """)

    row = db.execute(sql, {"var": var, "label": label, "unit": unit}).mappings().first()
    return row["fc"]
# app/main.py
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Query
import json
import joblib
from sklearn.preprocessing import StandardScaler
import numpy as np

from pydantic import BaseModel



from app.db import get_db
from app.schemas import ParcelaUpdate, ParcelaIn
import app.crud


app = FastAPI(title="Banano GIS API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Cargar el modelo
model = joblib.load('Modelo/modelo_fertilizacion_optimizado.pkl')

# Cargar el escalador (si es necesario, según cómo lo configuramos en el entrenamiento)

scaler = StandardScaler()

@app.get("/suelos/lote/{lote}", tags=["Suelos"])
def suelos_hist_lote(lote: str, db: Session = Depends(get_db)):
    sql = text("""
      SELECT
        p.lote,
        p.nombre,
        p.has,
        s.ph, s.mo, s.nh4, s.p, s.k, s.ca, s.mg,
        s.zn, s.cu, s.fe, s.mn, s.b, s.s,
        b.produccion_total_lbs,  -- Aquí estamos accediendo a la tabla de producción
        b.racimos_ha, b.peso_racimo
      FROM parcelas p
      LEFT JOIN banano_suelos s ON s.lote = p.lote
      LEFT JOIN banano_produccion_mensual b ON b.lote = p.lote
      WHERE p.lote = :lote
      ORDER BY b.anio DESC, b.mes DESC;
    """)
    rows = db.execute(sql, {"lote": lote.strip()}).mappings().all()

    if not rows:
        raise HTTPException(status_code=404, detail="No se encontraron datos para el lote")

    return {"lote": lote.strip(), "total": len(rows), "items": rows}



@app.post("/parcelas", tags=["Parcelas"])
def upsert_parcela(payload: ParcelaIn, db: Session = Depends(get_db)):
    # GeoJSON -> PostGIS
    try:
        geom_json = json.dumps(payload.geom)
    except Exception:
        raise HTTPException(status_code=400, detail="geom debe ser un objeto GeoJSON válido")

    sql = text("""
    INSERT INTO parcelas (lote, nombre, geom)
    VALUES (:lote, :nombre, ST_SetSRID(ST_GeomFromGeoJSON(:geom), 4326))
    ON CONFLICT (lote) DO UPDATE SET
      nombre = COALESCE(EXCLUDED.nombre, parcelas.nombre),
      geom = EXCLUDED.geom
    RETURNING lote;
    """)

    row = db.execute(sql, {
        "lote": payload.lote.strip(),
        "nombre": payload.nombre,
        "geom": geom_json
    }).mappings().first()

    if not row:
        raise HTTPException(status_code=400, detail="No se pudo guardar la parcela")

    db.commit()
    return {"ok": True, "lote": row["lote"]}


@app.patch("/parcelas/{lote}", tags=["Parcelas"])
def editar_parcela(lote: str, payload: ParcelaUpdate, db: Session = Depends(get_db)):
    row = crud.update_parcela(db, lote.strip(), payload.nombre, payload.geom)
    if not row:
        raise HTTPException(status_code=404, detail="Lote no encontrado")
    return {
        "type": "Feature",
        "geometry": row["geom"],
        "properties": {
            "parcela_id": str(row["parcela_id"]),
            "lote": row["lote"],
            "nombre": row["nombre"],
            "has": float(row["has"]) if row["has"] is not None else None
        }
    }


@app.get("/mapa/lotes", tags=["Mapa"])
def mapa_lotes(anio: int, mes: int, variable: str = "produccion_lbs_ha", db: Session = Depends(get_db)):
    try:
        fc = crud.get_mapa_lotes_geojson(db, anio, mes, variable)
        return fc
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ✅ Para el MAPA: devuelve SOLO parcelas con geometría (las que realmente se pueden dibujar)
@app.get("/parcelas", tags=["Parcelas"])
def listar_parcelas_con_geom(db: Session = Depends(get_db)):
    sql = text("""
      SELECT jsonb_build_object(
        'type','FeatureCollection',
        'features', COALESCE(jsonb_agg(
          jsonb_build_object(
            'type','Feature',
            'geometry', ST_AsGeoJSON(geom)::jsonb,
            'properties', jsonb_build_object(
              'parcela_id', parcela_id,
              'lote', lote,
              'nombre', nombre,
              'has', has
            )
          )
        ), '[]'::jsonb)
      ) AS fc
      FROM parcelas
      WHERE geom IS NOT NULL;
    """)
    row = db.execute(sql).mappings().first()
    return row["fc"]


# ✅ Por si quieres ver TODAS, aunque no tengan geometría (para debugging)
@app.get("/parcelas/todas", tags=["Parcelas"])
def listar_todas_parcelas(db: Session = Depends(get_db)):
    sql = text("""
      SELECT lote, nombre, has, (geom IS NOT NULL) AS tiene_geom
      FROM parcelas
      ORDER BY lote::int;
    """)
    rows = db.execute(sql).mappings().all()
    return {"total": len(rows), "items": rows}

@app.get("/suelos/resumen", tags=["Suelos"])
def suelos_resumen(db: Session = Depends(get_db)):
    sql = text("""
      SELECT
        p.lote,
        p.nombre,
        p.has,

        -- principales
        s.ph,
        s.mo            AS mat_org_pct,
        s.nh4           AS n_suelo_ppm,
        s.p             AS p_suelo_ppm,
        s.k             AS k_suelo_meq_100g,

        -- textura
        s.arena,
        s.limo,
        s.arcilla,

        -- micronutrientes
        s.zn            AS zn_suelo_ppm,
        s.cu            AS cu_suelo_ppm,
        s.fe            AS fe_suelo_ppm,
        s.mn            AS mn_suelo_ppm,
        s.b             AS b_suelo_ppm,
        s.s             AS s_suelo_ppm,
        s.ca            AS ca_suelo,
        s.mg            AS mg_suelo,

        -- alertas (CORRECTAMENTE ubicadas)
        (
          (CASE WHEN s.ph IS NOT NULL AND (s.ph < 5.5 OR s.ph > 7.2) THEN 1 ELSE 0 END) +
          (CASE WHEN s.mo IS NOT NULL AND s.mo < 2.0 THEN 1 ELSE 0 END) +
          (CASE WHEN s.p  IS NOT NULL AND s.p  < 10  THEN 1 ELSE 0 END) +
          (CASE WHEN s.k  IS NOT NULL AND s.k  < 0.4 THEN 1 ELSE 0 END)
        ) AS alertas_count,

        jsonb_build_object(
          'ph_fuera_rango', (s.ph IS NOT NULL AND (s.ph < 5.5 OR s.ph > 7.2)),
          'mo_baja',        (s.mo IS NOT NULL AND s.mo < 2.0),
          'p_bajo',         (s.p  IS NOT NULL AND s.p < 10),
          'k_bajo',         (s.k  IS NOT NULL AND s.k < 0.4),
          'textura_liviana',(s.arena IS NOT NULL AND s.arena > 60)
        ) AS alertas_detalle,

        -- sin fechas
        NULL::int AS anio,
        NULL::int AS mes

      FROM parcelas p
      LEFT JOIN banano_suelos s ON s.lote = p.lote
      ORDER BY p.lote::int;
    """)
    rows = db.execute(sql).mappings().all()
    return {"total": len(rows), "items": rows}


@app.get("/suelos/lote/{lote}", tags=["Suelos"])
def suelos_hist_lote(lote: str, db: Session = Depends(get_db)):
    sql = text("""
      SELECT
        analisis_id, lote, anio, mes,
        ph, mat_org_pct, n_suelo_ppm, p_suelo_ppm, k_suelo_meq_100g,
        ca_suelo_meq_100g, mg_suelo_meq_100g, s_suelo_ppm,
        zn_suelo_ppm, cu_suelo_ppm, fe_suelo_ppm, mn_suelo_ppm, b_suelo_ppm,
        n_foliar_pct, p_foliar_pct, k_foliar_pct, ca_foliar_pct, mg_foliar_pct, s_foliar_pct,
        zn_foliar_ppm, cu_foliar_ppm, fe_foliar_ppm, mn_foliar_ppm, b_foliar_ppm,
        raices_totales, raices_funcionales, pct_raices_funcionales,
        r_similis, helicotilenchus, meloidogine, t_nematodos,
        pct_racimos_grandes, pct_racimos_medianos, pct_racimos_pequenos,
        peso_promedio_racimo_lbs,
        creado_en
      FROM banano_analisis
      WHERE lote = :lote
      ORDER BY anio DESC, mes DESC;
    """)
    rows = db.execute(sql, {"lote": lote.strip()}).mappings().all()
    return {"lote": lote.strip(), "total": len(rows), "items": rows}

@app.get("/mapa/suelos", tags=["Mapa"])
def mapa_suelos(db: Session = Depends(get_db)):
    sql = text("""
      WITH base AS (
        SELECT
          p.lote,
          p.nombre,
          p.has,
          p.geom,
          s.ph, s.mo, s.p, s.k,
          s.arena, s.limo, s.arcilla,
          s.zn, s.cu, s.fe, s.mn, s.b, s.s, s.ca, s.mg,

          (
            (CASE WHEN s.ph IS NOT NULL AND (s.ph < 5.5 OR s.ph > 7.2) THEN 1 ELSE 0 END) +
            (CASE WHEN s.mo IS NOT NULL AND s.mo < 2.0 THEN 1 ELSE 0 END) +
            (CASE WHEN s.p  IS NOT NULL AND s.p  < 10  THEN 1 ELSE 0 END) +
            (CASE WHEN s.k  IS NOT NULL AND s.k  < 0.4 THEN 1 ELSE 0 END)
          ) AS alertas_count,

          jsonb_build_object(
            'ph_fuera_rango', (s.ph IS NOT NULL AND (s.ph < 5.5 OR s.ph > 7.2)),
            'mo_baja',        (s.mo IS NOT NULL AND s.mo < 2.0),
            'p_bajo',         (s.p  IS NOT NULL AND s.p < 10),
            'k_bajo',         (s.k  IS NOT NULL AND s.k < 0.4),
            'textura_liviana',(s.arena IS NOT NULL AND s.arena > 60)
          ) AS alertas_detalle
        FROM parcelas p
        LEFT JOIN banano_suelos s ON s.lote = p.lote
        WHERE p.geom IS NOT NULL
      )
      SELECT jsonb_build_object(
        'type','FeatureCollection',
        'features', COALESCE(jsonb_agg(
          jsonb_build_object(
            'type','Feature',
            'geometry', ST_AsGeoJSON(geom)::jsonb,
            'properties', jsonb_build_object(
              'lote', lote,
              'nombre', nombre,
              'has', has,
              'ph', ph,
              'mo', mo,
              'p', p,
              'k', k,
              'arena', arena, 'limo', limo, 'arcilla', arcilla,
              'alertas_count', alertas_count,
              'alertas_detalle', alertas_detalle
            )
          )
        ), '[]'::jsonb)
      ) AS fc
      FROM base;
    """)
    row = db.execute(sql).mappings().first()
    return row["fc"]

@app.get("/parcelas/ubicaciones", tags=["Parcelas"])
def parcelas_ubicaciones(db: Session = Depends(get_db)):
    sql = text("""
      SELECT
        lote,
        nombre,
        has,
        (geom IS NOT NULL) AS tiene_geom,
        CASE
          WHEN geom IS NOT NULL THEN ST_X(ST_Centroid(geom))
          ELSE NULL
        END AS lon,
        CASE
          WHEN geom IS NOT NULL THEN ST_Y(ST_Centroid(geom))
          ELSE NULL
        END AS lat
      FROM parcelas
      ORDER BY lote::int;
    """)
    rows = db.execute(sql).mappings().all()
    return {"total": len(rows), "items": rows}

@app.get("/mapa/suelos", tags=["Mapa"])
def mapa_suelos_var(
    var: str = Query("ph", description="Variable de suelo a mapear"),
    db: Session = Depends(get_db)
):
    # ✅ Lista permitida (variable -> (expresión SQL, etiqueta, unidad))
    VARS = {
        "ph":      ("s.ph", "pH", ""),
        "mo":      ("s.mo", "Materia orgánica", "%"),
        "arena":   ("s.arena", "Arena", "%"),
        "limo":    ("s.limo", "Limo", "%"),
        "arcilla": ("s.arcilla", "Arcilla", "%"),
        "nh4":     ("s.nh4", "NH4", "ppm"),
        "p":       ("s.p", "Fósforo (P)", "ppm"),
        "k":       ("s.k", "Potasio (K)", "meq/100g"),
        "ca":      ("s.ca", "Calcio (Ca)", "meq/100g"),
        "mg":      ("s.mg", "Magnesio (Mg)", "meq/100g"),
        "zn":      ("s.zn", "Zinc (Zn)", "ppm"),
        "cu":      ("s.cu", "Cobre (Cu)", "ppm"),
        "fe":      ("s.fe", "Hierro (Fe)", "ppm"),
        "mn":      ("s.mn", "Manganeso (Mn)", "ppm"),
        "b":       ("s.b", "Boro (B)", "ppm"),
        "s":       ("s.s", "Azufre (S)", "ppm"),
        "alertas": ("""
            (
              (CASE WHEN s.ph IS NOT NULL AND (s.ph < 5.5 OR s.ph > 7.2) THEN 1 ELSE 0 END) +
              (CASE WHEN s.mo IS NOT NULL AND s.mo < 2.0 THEN 1 ELSE 0 END) +
              (CASE WHEN s.p  IS NOT NULL AND s.p  < 10  THEN 1 ELSE 0 END) +
              (CASE WHEN s.k  IS NOT NULL AND s.k  < 0.4 THEN 1 ELSE 0 END)
            )
        """, "Alertas", "count"),
    }

    if var not in VARS:
        raise HTTPException(status_code=400, detail=f"Variable inválida: {var}")

    expr, label, unit = VARS[var]

    sql = text(f"""
      WITH base AS (
        SELECT
          p.lote,
          p.nombre,
          p.has,
          p.geom,

          -- valores base
          s.ph, s.mo, s.arena, s.limo, s.arcilla, s.nh4,
          s.p, s.k, s.ca, s.mg, s.zn, s.cu, s.fe, s.mn, s.b, s.s,

          -- variable seleccionada
          ({expr})::numeric AS valor,

          -- alertas (para popup)
          (
            (CASE WHEN s.ph IS NOT NULL AND (s.ph < 5.5 OR s.ph > 7.2) THEN 1 ELSE 0 END) +
            (CASE WHEN s.mo IS NOT NULL AND s.mo < 2.0 THEN 1 ELSE 0 END) +
            (CASE WHEN s.p  IS NOT NULL AND s.p  < 10  THEN 1 ELSE 0 END) +
            (CASE WHEN s.k  IS NOT NULL AND s.k  < 0.4 THEN 1 ELSE 0 END)
          ) AS alertas_count,

          jsonb_build_object(
            'ph_fuera_rango', (s.ph IS NOT NULL AND (s.ph < 5.5 OR s.ph > 7.2)),
            'mo_baja',        (s.mo IS NOT NULL AND s.mo < 2.0),
            'p_bajo',         (s.p  IS NOT NULL AND s.p < 10),
            'k_bajo',         (s.k  IS NOT NULL AND s.k < 0.4),
            'textura_liviana',(s.arena IS NOT NULL AND s.arena > 60)
          ) AS alertas_detalle

        FROM parcelas p
        LEFT JOIN banano_suelos s ON s.lote = p.lote
        WHERE p.geom IS NOT NULL
      )
      SELECT jsonb_build_object(
        'type','FeatureCollection',
        'meta', jsonb_build_object(
          'var', :var,
          'label', :label,
          'unit', :unit
        ),
        'features', COALESCE(jsonb_agg(
          jsonb_build_object(
            'type','Feature',
            'geometry', ST_AsGeoJSON(geom)::jsonb,
            'properties', jsonb_build_object(
              'lote', lote,
              'nombre', nombre,
              'has', has,

              'valor', valor,
              'var', :var,
              'label', :label,
              'unit', :unit,

              'ph', ph, 'mo', mo,
              'arena', arena, 'limo', limo, 'arcilla', arcilla,
              'nh4', nh4, 'p', p, 'k', k,
              'ca', ca, 'mg', mg,
              'zn', zn, 'cu', cu, 'fe', fe, 'mn', mn, 'b', b, 's', s,

              'alertas_count', alertas_count,
              'alertas_detalle', alertas_detalle
            )
          )
        ), '[]'::jsonb)
      ) AS fc
      FROM base;
    """)

    row = db.execute(sql, {"var": var, "label": label, "unit": unit}).mappings().first()
    return row["fc"]
