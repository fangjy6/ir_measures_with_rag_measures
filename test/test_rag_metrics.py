import ir_measures
from ir_measures import EM, F1
# from ir_measures.providers import RAGProvider
import pandas as pd

# the gold answer can either be a str or List[str]. However, F1 score will only use the first answer to calculate the score
qrels = {"q1": "University of Glasgow", "q2": "Glasgow, Scotland", "q3": ["multiple answer test", "answer1"]}
qrels_df = pd.DataFrame.from_dict({"qid": ["q1", "q2", "q3"], "gold_answer": ["University of Glasgow", "Glasgow, Scotland", ["multiple_answer_test", "answer1"]]})

# the predicted answer must be a single str 
dict_run = {"q1": "University of Glasgow", "q2": "Scotland", "q3": "answer2"}
df_run = pd.DataFrame.from_dict({"qid": ["q1", "q2", "q3"], "pred_answer": ["University of Glasgow", "Scotland", "answer2"]})

# use dicts as inputs 
metrics = ir_measures.calc_aggregate([EM, F1], qrels, dict_run)
print(metrics)

# use dataframe as inputs 
metrics = ir_measures.calc_aggregate([EM, F1], qrels_df, df_run)
print(metrics)

# use both dict and dataframe as inputs 
metrics = ir_measures.calc_aggregate([EM, F1], qrels, df_run)
print(metrics)

# by query 
for m in ir_measures.iter_calc([EM, F1], qrels_df, df_run):
    print(m)