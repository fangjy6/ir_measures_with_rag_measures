from typing import Iterator
import ir_measures
from ir_measures import providers, measures, Metric
from ir_measures.bin import rag_eval
from functools import partial
import pandas as pd 
import warnings

class RAGProvider(providers.Provider):

    NAME = "rag_eval"
    SUPPORTED_MEASURES = [measures._EM(), measures._F1()]

    def _evaluator(self, measures, qrels):

        if isinstance(qrels, pd.DataFrame):
            qrels = self.convert_dataframework_to_dict(qrels, "qid", "gold_answer")
        
        invocations = []
        for measure in measures:
            if measure.NAME == "EM":
                ems_qrels = self.check_ems_qrels(qrels)
                invocations.append((measure, ems_qrels, rag_eval.ems))
            elif measure.NAME == "F1":
                f1_qrels = self.check_f1_score_qrels(qrels)
                invocations.append((measure, f1_qrels, rag_eval.f1_score))
            else:
                raise ValueError(f'unsupported measure {measure}')
        
        return RAGEvaluator(measures, qrels, invocations)
    
    def convert_dataframework_to_dict(self, df: pd.DataFrame, key_col: str, value_col: str):
        columns = df.columns
        missing_columns = {key_col, value_col} - set(columns)
        if missing_columns:
            raise ValueError(f"DataFrame missing columns: {list(missing_columns)} (found {list(columns)})")
        result = {}
        for item in df.to_dict(orient='records'):
            result[item[key_col]] = item[value_col]
        return result
    
    def check_ems_qrels(self, qrels):

        new_qrels = {}
        for qid, answers in qrels.items():
            if isinstance(answers, str):
                answers = [answers]
            new_qrels[qid] = answers
        return new_qrels

    def check_f1_score_qrels(self, qrels):

        has_multiple_answers = False
        new_qrels = {}
        for qid, answer in qrels.items():
            if isinstance(answer, list):
                if len(answer) > 1:
                    has_multiple_answers = True
                answer = answer[0]
            new_qrels[qid] = answer
        if has_multiple_answers:
            warnings.warn("There are multiple answers for a question. Only the first answer will be used to calculate F1 score!")
        return new_qrels
    

class RAGEvaluator(providers.Evaluator):

    def __init__(self, measures, qrels, invocations):
        self.query_ids = {query_id for query_id, _ in qrels.items()}
        super().__init__(measures, self.query_ids)
        self.qrels = qrels
        self.invocations = invocations
    
    def _iter_calc(self, run):
        run = self.convert_run_format(run)
        self.check_run_qids(run)
        for measure, qrels, eval_function in self.invocations:
            for qid, pred in run.items():
                value = eval_function(pred, qrels[qid])
                yield Metric(query_id=qid, measure=measure, value=value)
    
    def convert_run_format(self, run):
        """
        run: {qid: pred_answer}
        """
        if isinstance(run, dict):
            return run
        elif isinstance(run, pd.DataFrame):
            missing_cols = {"qid", "pred_answer"} - set(run.columns) 
            if missing_cols:
                raise ValueError(f"DataFrame missing columns: {list(missing_cols)} (found {list(run.columns)})")
            run_as_dict = {}
            for one_run in run.itertuples():
                run_as_dict[one_run.qid] = one_run.pred_answer
            return run_as_dict
        else:
            raise ValueError(f"type {type(run)} of run is not supported!")

    def check_run_qids(self, run):

        missing_ground_truth_qids = []
        for qid, pred in run.items():
            if qid not in self.query_ids:
                missing_ground_truth_qids.append(qid)
                break
        if missing_ground_truth_qids:
            raise ValueError(f"Missing gold answers for qids: {missing_ground_truth_qids}")

providers.register(RAGProvider())