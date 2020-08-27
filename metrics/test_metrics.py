import metrics
from copy import copy
from random import shuffle
from metrics import evaluationParams
import rrc_evaluation_funcs_1_1 as rrc_evaluation_funcs


def gt_zip_to_list(gtFilePath='./gt.zip'):
    """Convert the gt.zip from IDCAR into our format."""
    gt = rrc_evaluation_funcs.load_zip_file(gtFilePath, evaluationParams['GT_SAMPLE_NAME_2_ID'])

    points = []
    for resFile in gt:
        gtFile = rrc_evaluation_funcs.decode_utf8(gt[resFile])
        pointsList,_,transcriptionsList = rrc_evaluation_funcs.get_tl_line_values_from_file_contents(gtFile,evaluationParams['CRLF'],evaluationParams['LTRB'],True,False)
        points.append(pointsList)

    return points

pts = gt_zip_to_list()

results = metrics.get_metrics(pts, pts, scores=[])
print(results)

res = [pts[i] if i < 400 else pts[0] for i in range(500)]
results = metrics.get_metrics(pts, res, scores=[])
print(results)

res = copy(pts)
shuffle(pts)
shuffle(res)
results = metrics.get_metrics(pts, res, scores=[])
print(results)
