# app/main.py
from fastapi import FastAPI, Depends, HTTPException, Query, Request, Response
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


@app.middleware("http")
async def add_cors_headers(request: Request, call_next):
    if request.method == "OPTIONS":
        return Response(
            status_code=200,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "*",
                "Access-Control-Allow-Headers": "*",
            }
        )
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar el modelo
model = joblib.load('Modelo/modelo_fertilizacion_optimizado.pkl')

# Escalador
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
        input_data.ph, input_data.mo, input_data.nh4, input_data.p,
        input_data.k, input_data.ca, input_data.mg, input_data.zn,
        input_data.cu, input_data.fe, input_data.mn, input_data.b,
        input_data.s, input_data.racimos_ha, input_data.peso_racimo, input_data.has
    ]])
    input_array_scaled = scaler.fit_transform(input_array)
    prediccion = model.predict(input_array_scaled)
    return {"prediccion_fertilizante": np.expm1(prediccion[0])}


@app.post("/parcelas", tags=["Parcelas"])
def upsert_parcela(payload: ParcelaIn, db: Session = Depends(get_db)):
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
        p.lote, p.nombre, p.has,
        s.ph,
        s.mo            AS mat_org_pct,
        s.nh4           AS n_suelo_ppm,
        s.p             AS p_suelo_ppm,
        s.k             AS k_suelo_meq_100g,
        s.arena, s.limo, s.arcilla,
        s.zn            AS zn_suelo_ppm,
        s.cu            AS cu_suelo_ppm,
        s.fe            AS fe_suelo_ppm,
        s.mn            AS mn_suelo_ppm,
        s.b             AS b_suelo_ppm,
        s.s             AS s_suelo_ppm,
        s.ca            AS ca_suelo,
        s.mg            AS mg_suelo,
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
        p.lote, p.nombre, p.has,
        s.ph, s.mo, s.nh4, s.p, s.k, s.ca, s.mg,
        s.zn, s.cu, s.fe, s.mn, s.b, s.s,
        b.produccion_total_lbs,
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


@app.get("/parcelas/ubicaciones", tags=["Parcelas"])
def parcelas_ubicaciones(db: Session = Depends(get_db)):
    sql = text("""
      SELECT
        lote, nombre, has,
        (geom IS NOT NULL) AS tiene_geom,
        CASE WHEN geom IS NOT NULL THEN ST_X(ST_Centroid(geom)) ELSE NULL END AS lon,
        CASE WHEN geom IS NOT NULL THEN ST_Y(ST_Centroid(geom)) ELSE NULL END AS lat
      FROM parcelas
      ORDER BY lote::int;
    """)
    rows = db.execute(sql).mappings().all()
    return {"total": len(rows), "items": rows}


@app.get("/mapa/suelos", tags=["Mapa"])
def mapa_suelos(
    var: str = Query("ph", description="Variable de suelo a mapear"),
    db: Session = Depends(get_db)
):
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
          p.lote, p.nombre, p.has, p.geom,
          s.ph, s.mo, s.arena, s.limo, s.arcilla, s.nh4,
          s.p, s.k, s.ca, s.mg, s.zn, s.cu, s.fe, s.mn, s.b, s.s,
          ({expr})::numeric AS valor,
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
        'meta', jsonb_build_object('var', :var, 'label', :label, 'unit', :unit),
        'features', COALESCE(jsonb_agg(
          jsonb_build_object(
            'type','Feature',
            'geometry', ST_AsGeoJSON(geom)::jsonb,
            'properties', jsonb_build_object(
              'lote', lote, 'nombre', nombre, 'has', has,
              'valor', valor, 'var', :var, 'label', :label, 'unit', :unit,
              'ph', ph, 'mo', mo,
              'arena', arena, 'limo', limo, 'arcilla', arcilla,
              'nh4', nh4, 'p', p, 'k', k, 'ca', ca, 'mg', mg,
              'zn', zn, 'cu', cu, 'fe', fe, 'mn', mn, 'b', b, 's', s,
              'alertas_count', alertas_count, 'alertas_detalle', alertas_detalle
            )
          )
        ), '[]'::jsonb)
      ) AS fc
      FROM base;
    """)

    row = db.execute(sql, {"var": var, "label": label, "unit": unit}).mappings().first()
    return row["fc"]