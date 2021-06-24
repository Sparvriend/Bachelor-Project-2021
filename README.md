# Author: Timo Wahl (S3812030)
# Bachelor-Project-2021
# Machine learning systems rationale analysis
# Daily Supervisor: C.C. Steging & Second Assesor: B. Verheij

# Abstract:
Bench-Capon (1993) showed that high accuracies on a classification task does not necessarily mean that the reasoning of the model making the prediction is sound. In that paper it was shown that in depth analysis of the decision making process of an MLP can indicate the discrepancies in the models understanding, or rationale, of the domain that was trained on. This paper first replicated the original paper by Bench-Capon and then extended that paper with a more in depth analysis of the domain, new machine learning systems and improved hyperparameter optimization. The MLP, Random Forest classifier and XGBoost classifier were all compared in their performance and rationales on the same domain. The conclusions from the paper by Bench-Capon were reaffirmed, with the primary conclusion from the extension being that the Random Forest and XGBoost classifiers are not capable of learning completely sound rationales for the welfare benefit domain, despite being more optimized and being trained on larger datasets.

# Branch explanation
There are two branches. The replication branch includes all the code that is needed to get data and test with the MLPs of the replication. The Research branch includes all the code that is needed to get data and test with the MLP, Random Forest and XGBoost for the extension. The code is commented for more clarity.

