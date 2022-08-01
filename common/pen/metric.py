from pathlib import Path
from typing import List, Dict

import numpy as np
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.eval import COCOEvalCap
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocotools.coco import COCO

BLEURT_PATH = Path(__file__).parent.parent.parent / 'resource' / 'bleurt'


class BLEURT:
    def __init__(self):
        import tensorflow as tf
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.set_logical_device_configuration(
                    gpus[0],
                    [tf.config.LogicalDeviceConfiguration(memory_limit=4096)])
            except RuntimeError as e:
                pass

        from bleurt.score import LengthBatchingBleurtScorer
        self.scorer = LengthBatchingBleurtScorer(str(BLEURT_PATH.absolute()))

    def method(self):
        return "BLEURT"

    def compute_score(self, gts, res):
        sentence_ids = sorted(gts.keys())

        hypothesis = [res[i][0] for i in sentence_ids]
        references = [gts[i] for i in sentence_ids]
        assert min(len(x) for x in references) == max(len(x) for x in references)

        ref_len = min(len(x) for x in references)
        scores = []
        for r in range(ref_len):
            ref_r = [x[r] for x in references]
            scores_r = self.scorer.score(references=ref_r, candidates=hypothesis)

            if len(scores) == 0:
                scores = scores_r
            else:
                scores = [max(pre, cur) for pre, cur in zip(scores, scores_r)]

        return float(np.mean(scores)), scores


class COCOEvalCapModified(COCOEvalCap):
    def evaluate(self):
        imgIds = self.params['image_id']
        # imgIds = self.coco.getImgIds()
        gts = {}
        res = {}
        for imgId in imgIds:
            gts[imgId] = self.coco.imgToAnns[imgId]
            res[imgId] = self.cocoRes.imgToAnns[imgId]

        # =================================================
        # Set up scorers
        # =================================================
        print('tokenization...')
        tokenizer = PTBTokenizer()
        gts = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)

        # *** Modification begin
        # =================================================
        # Set up scorers
        # =================================================
        print('setting up scorers...')

        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr")
            # SPICE is excluded (we don't have any images!)
        ]

        if BLEURT_PATH.exists():
            scorers.append((BLEURT(), "BLEURT"))
        # *** Modification end

        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            print('computing %s score...' % (scorer.method()))
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, gts.keys(), m)
                    print("%s: %0.3f" % (m, sc))
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, gts.keys(), method)
                print("%s: %0.3f" % (method, score))
        self.setEvalImgs()


def explanations_as_coco(items: Dict[str, List[str]]):
    coco = COCO()
    coco.dataset = {
        'annotations': [{'id': '%s_%s' % (key, did), 'image_id': key, 'caption': expl}
                        for key, expl_list in items.items()
                        for did, expl in enumerate(expl_list)],
        'images': [{'id': key} for key in items.keys()]
    }
    coco.createIndex()
    return coco


def compute_metrics(references: Dict[str, List[str]], hypothesis: Dict[str, List[str]]) -> Dict[str, float]:
    dataset = explanations_as_coco(references)
    results = explanations_as_coco(hypothesis)
    coco_eval = COCOEvalCapModified(dataset, results)
    coco_eval.evaluate()

    return coco_eval.eval
