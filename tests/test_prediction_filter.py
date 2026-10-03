from src.pi.prediction_filter import PredictionFilter
def test_votes_and_low_confidence():
 f=PredictionFilter(7,5,.8)
 assert f.update("wet",.9) is None
 for x in ["wet","wet","wet","wet","dry","dry"]: result=f.update(x,.9)
 assert result.label=="wet"
 assert PredictionFilter(3,2,.8).update("dry",.5) is None
