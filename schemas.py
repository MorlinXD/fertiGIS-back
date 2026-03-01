# app/schemas.py
from pydantic import BaseModel, Field
from typing import Any, Optional, Dict, List

class ParcelaCreate(BaseModel):
  lote: str = Field(..., min_length=1)
  nombre: Optional[str] = None
  geom: Dict[str, Any]  # GeoJSON geometry (Polygon o MultiPolygon)

class ParcelaUpdate(BaseModel):
  nombre: Optional[str] = None
  geom: Optional[Dict[str, Any]] = None  # si solo cambias nombre, geom puede ser None

class Feature(BaseModel):
  type: str = "Feature"
  geometry: Dict[str, Any]
  properties: Dict[str, Any]

class FeatureCollection(BaseModel):
  type: str = "FeatureCollection"
  features: List[Feature]

class ParcelaIn(BaseModel):
    lote: str
    nombre: str | None = None
    geom: dict  # GeoJSON geometry
