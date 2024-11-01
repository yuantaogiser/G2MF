ROAD_SELECT_EXPRESSION = "highway IN ('motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'residential', 'unclassified') or railway IN ('rail')"

CALCULATE_FIEDLD_EXPRESSION = """def getDis(x, y):
    if x in ['motorway', 'trunk']:
        return 40
    if x in ['primary', 'tertiary']:
        return 20
    if x in ['residential', 'secondary', 'unclassified']:
        return 10
    if y in ['rail']:
        return 40
    return 10"""

FIELD_TO_KEEP = ['osm_id',  'code', 'name', 'highway','railway']

PROJ_CRS = 3857
GEO_CRS = 4326
