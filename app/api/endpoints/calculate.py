from fastapi import HTTPException, APIRouter, Query

from pollinator_abundance.handler import pollinator_abundance_calculation

# Simulated database
mock_db = {
    1: {"name": "Plantation A", "location": "Region 1"},
    2: {"name": "Plantation B", "location": "Region 2"},
    3: {"name": "Plantation C", "location": "Region 3"},
}


def check_plantation_exists(plantation_id: int):
    if plantation_id not in mock_db.keys():
        raise HTTPException(status_code=404, detail=f"Plantation ID {plantation_id} not found")
    return mock_db[plantation_id]


class CalculateHandler:
    def __init__(self):
        self.router = APIRouter()
        self.router.add_api_route("/calculate", self.get, methods=["GET"])

    async def get(self, plantation_id: int = Query(description="Plantation ID", gt=0, lt=1000000),
                  plantations_polygons_id: int = Query(description="Plantation polygons ID", gt=0, lt=1000000),
                  resolution: str = Query("low", enum=["low", "medium", "high"], description="Requested resolution"),
                  ca_id: int = Query(description="CA ID", gt=0, lt=1000000),
                  roi_id: int = Query(description="ROI ID", gt=0, lt=1000000),
                  override_bee: bool = Query(True, description="Override bee flag"),
                  how: str = Query("local", enum=["local", "global"], description="Processing method")):
        # Simulated plantation_id check
        check_plantation_exists(plantation_id)

        result = pollinator_abundance_calculation(plantation_id=plantation_id,
                                                  plantations_polygons_id=plantations_polygons_id,
                                                  resolution=resolution,
                                                  ca_id=ca_id,
                                                  roi_id=roi_id,
                                                  override_bee=override_bee,
                                                  how=how)
        return result["result_values"]
