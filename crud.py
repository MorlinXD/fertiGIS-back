# app/crud.py
from sqlalchemy import text
from sqlalchemy.orm import Session

def upsert_parcela(db: Session, lote: str, nombre: str | None, geom_geojson: dict):
  sql = text("""
    INSERT INTO parcelas (lote, nombre, geom)
    VALUES (:lote, :nombre,
      ST_Multi(
        ST_SetSRID(
          ST_GeomFromGeoJSON(:geom_json),
          4326
        )
      )
    )
    ON CONFLICT (lote)
    DO UPDATE SET
      nombre = COALESCE(EXCLUDED.nombre, parcelas.nombre),
      geom = COALESCE(EXCLUDED.geom, parcelas.geom),
      actualizado_en = now()
    RETURNING parcela_id, lote, nombre, has,
      ST_AsGeoJSON(geom)::json AS geom;
  """)
  row = db.execute(sql, {"lote": lote, "nombre": nombre, "geom_json": str(geom_geojson).replace("'", '"')}).mappings().first()
  db.commit()
  return row

def update_parcela(db: Session, lote: str, nombre: str | None, geom_geojson: dict | None):
  if geom_geojson is None:
    sql = text("""
      UPDATE parcelas
      SET nombre = COALESCE(:nombre, nombre),
          actualizado_en = now()
      WHERE lote = :lote
      RETURNING parcela_id, lote, nombre, has,
        ST_AsGeoJSON(geom)::json AS geom;
    """)
    row = db.execute(sql, {"lote": lote, "nombre": nombre}).mappings().first()
    db.commit()
    return row

  sql = text("""
    UPDATE parcelas
    SET nombre = COALESCE(:nombre, nombre),
        geom = ST_Multi(ST_SetSRID(ST_GeomFromGeoJSON(:geom_json), 4326)),
        actualizado_en = now()
    WHERE lote = :lote
    RETURNING parcela_id, lote, nombre, has,
      ST_AsGeoJSON(geom)::json AS geom;
  """)
  row = db.execute(sql, {"lote": lote, "nombre": nombre, "geom_json": str(geom_geojson).replace("'", '"')}).mappings().first()
  db.commit()
  return row

ALLOWED_VARS = {
  "produccion_total_lbs": "produccion_total_lbs",
  "produccion_lbs_ha": "produccion_lbs_ha",
  "ph": "ph",
  "mat_org_pct": "mat_org_pct",
  "p_suelo_ppm": "p_suelo_ppm",
  "k_suelo_meq_100g": "k_suelo_meq_100g",
  "t_nematodos": "t_nematodos",
  "peso_promedio_racimo_lbs": "peso_promedio_racimo_lbs",
}

def get_mapa_lotes_geojson(db: Session, anio: int, mes: int, variable: str):
  if variable not in ALLOWED_VARS:
    raise ValueError(f"Variable inválida. Usa una de: {list(ALLOWED_VARS.keys())}")

  col = ALLOWED_VARS[variable]

  # La vista vw_mapa_banano_lote_mes ya une parcelas + análisis + producción
  sql = text(f"""
    SELECT jsonb_build_object(
      'type', 'FeatureCollection',
      'features', COALESCE(jsonb_agg(
        jsonb_build_object(
          'type', 'Feature',
          'geometry', ST_AsGeoJSON(geom)::jsonb,
          'properties', jsonb_build_object(
            'lote', lote,
            'nombre', nombre,
            'has', has,
            'anio', anio,
            'mes', mes,
            'value', {col}
          )
        )
      ), '[]'::jsonb)
    ) AS fc
    FROM vw_mapa_banano_lote_mes
    WHERE anio = :anio AND mes = :mes;
  """)
  row = db.execute(sql, {"anio": anio, "mes": mes}).mappings().first()
  return row["fc"]
