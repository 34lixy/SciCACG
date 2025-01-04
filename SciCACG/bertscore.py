from bert_score import score

cand = ["I have an apple.", "I am Lucky."]
ref = ["I have a pen.", "I am Lucy."]
P, R, F1 = score(cand, ref, lang="en", model_type="facebook/bart-base")
bertscore = score(cand, ref, lang="en", model_type="facebook/bart-base")
f1_percentage = round(F1.mean().item() * 100, 2)

print('p:', P.mean().item(), 'r:', R.mean().item(), 'f1:', F1.mean().item())
print("bert Score: {:.2f}%".format(f1_percentage))
print('p:', bertscore['p'].mean().item(), 'r:', bertscore['r'].mean().item(), 'f1:', bertscore['f1'].mean().item())
