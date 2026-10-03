from src.common.mapping import map_label
def test_mapping():
 assert map_label("BIODEGRADABLE")=="wet"; assert map_label("organic")=="wet"; assert map_label("PAPER")=="dry"; assert map_label("plastic")=="dry"; assert map_label("Hazardous")=="reject"; assert map_label("e-waste")=="reject"
