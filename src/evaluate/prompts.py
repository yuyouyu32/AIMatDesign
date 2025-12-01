Rules = """- Compositions with a calculated average atomic size difference greater than 12% across all constituent elements reward high, as they are more likely to form amorphous structures.
- Alloys with five or more principal elements, indicating a high mixing entropy, reward highest.
- Compositions with a majority of element pairs having a negative heat of mixing reward higher, as they are more likely to form a stable amorphous phase.
- Compositions with a high modulus contrast among constituents, particularly with elements known for high Young's modulus, e.g., Ti and some rare earths like Gd, reward higher.
- Alloys with a Tg/Tl ratio closer to or greater than 0.6 reward highest, as they demonstrate better thermal stability.
- Compositions with a balanced ratio of late transition metals, particularly when combined with Zr or Ti, reward higher, as these are conducive to forming amorphous structures.
- Alloys with a greater variety of elements, especially those that introduce complexity without significantly increasing the tendency for crystalline phase formations, reward higher.
"""

EvalueSystem = "You will act as an expert in materials science, specializing in bulk metallic glass (BMG). Your task is to analyze provided data points, which are based on specific screening rules for BMGs components."

EvalueUser = """Firstly, understand and apply the following screening rules defined as RULE. These rules are crucial for evaluating the potential of each BMG component.I will also provide you with performance data for Real BMGs similar to the data in the DATA, which are actually measured and can be used as a reference for your evaluation.

Secondly, evaluate each data point in DATA by assigning a reward from -1 to 1, indicating its suitability for experimental validation in bulk metallic glass (BMG). A reward of 1 signifies high relevance. Include a short explanation for each reward, focusing on pertinent scientific concepts and considering the complexities and potential results of the experimental process in BMGs.

### RULE:
{rule}

### Real BMGs:
{sim}

### DATA:
{data}

Ensure the output for each evaluated data point is formatted as follows, data calculations can be analyzed if necessary. The format of the rating needs to be output in the following format, additional calculations are allowed, but the rating information needs to be wrapped in `{{}}`:
{{
    "reward": [assigned value],
    "reason": [brief overview explaining the assigned value]
}}"""

KBRPrompt = """你是一位在材料科学领域具有深厚造诣的专家，尤其对大块金属玻璃（BMG）的成分、性能和实验验证过程有着丰富的研究经验。你能够精准地运用科学原理和实验数据，对BMG成分的潜力进行客观评估。
你具备材料科学的基础理论知识，熟悉大块金属玻璃的制备方法、性能特点以及实验验证的复杂性。你能够根据提供的筛选规则和参考数据，对每个数据点进行科学合理的评估，并给出具有指导意义的奖励值。

现在你需要对特定的大块金属玻璃（BMG）成分进行评估，以确定其是否适合进行实验验证。请你参考下面提供筛选规则（RULE），以及实际测量的类似BMG的性能数据（Real BMGs），作为评估的参考。
你需要对每个数据点进行评估，并给出一个介于-1到1之间的奖励值，以指导强化学习（RL）模型的Knowledge-Base Reward。请你输出完整的推理过程，确保评估的科学性和准确性，你可以参考下面的推理流程：
1. 仔细阅读并理解提供的筛选规则（RULE），明确评估的关键指标和要求。
2. 对比参考数据（Real BMGs），分析其性能特点和实验验证结果，作为评估的基准。
3. 针对提供的数据点（DATA），根据筛选规则和参考数据进行综合评估，给出奖励值，并简要说明理由。

### RULE:
{rule}

### 相似的真实BMGs:
{sim}

### DATA:
{data}

评估过程中必须严格遵循提供的筛选规则（RULE），并结合实际测量的参考数据（Real BMGs）进行对比分析。
奖励值的范围限定在-1到1之间，其中1表示该数据点符合BMGs相关知识具有高的实验价值，赋予高reward；-1表示该数据点违背了RULE中的相关知识，赋予低reward。
最后你需要输出该### Data的评估结果，格式如下：
{{
    "reward": Data点的奖励值, [-1, 1], 保留两位小数
    "reason": brief overview explaining the assigned value
}}"
评估理由应简洁明了，突出关键的科学概念和实验验证的潜在结果。

现在请你开始评估数据点（DATA），并给出奖励值和评估理由。请注意，最终评估结果需要以JSON格式输出，确保格式正确。"""


KBRPrompt = """
You are an expert in materials science with extensive experience in Bulk Metallic Glass (BMG) composition, performance, and experimental validation. 
You can objectively assess the potential of BMG compositions using scientific principles and experimental data.

Given the following selection criteria (RULE) and performance data of similar BMGs (Similar Real BMGs), 
evaluate the provided data point (DATA) to determine its suitability for experimental validation. 
Assign a reward value between -1 and 1 to guide the reinforcement learning (RL) model's Knowledge-Base Reward. 

Provide a detailed reasoning process to ensure scientific accuracy:
1. Review and understand the selection criteria (RULE), identifying key indicators and requirements.
2. Compare with similar BMGs (Similar Real BMGs), analyzing performance characteristics and experimental outcomes as benchmarks.
3. Evaluate the provided data point (DATA) against the selection criteria and reference data, assigning a reward value and justifying your reasoning.

### RULE:
{rule}

### Similar Real BMGs:
{sim}

### DATA:
{data}

The reward value should range from -1 to 1, where 1 indicates high experimental value and alignment with BMG knowledge, and -1 indicates significant deviation from the criteria. 
Output the evaluation result in the following format:
{
    "reward": Data point's reward value, [-1, 1], rounded to two decimal places,
    "reason": "Brief explanation of the assigned value, highlighting key scientific concepts and potential experimental outcomes."
}
Now please start evaluating the data points (DATA) and give the award value and reason for the evaluation. 
Please note that the final evaluation results need to be output in JSON format to ensure that the format is correct.
"""

VarAMRPrompt = """你是大块金属玻璃（BMG）领域机器学习建模方面的专家，具有深入的材料成分、性能以及机器学习在材料科学中的应用知识和丰富的实践经验。你能够精准地分析当前模型存在的问题，并通过特征工程优化模型性能。

目前，Guiding Model（回归模型）在预测相似的 BMG {composition} 的 {performance} 时，预测方差较大（{pred_var}），模型的预测结果不够稳定。为了提高模型的预测性能，你需要从提供的候选特征中选择 1-3 个新特征，用于重新训练 Guiding Model（回归模型），帮助减少在预测相似 BMG 成分的 {performance} 时的预测方差。

请根据以下步骤进行特征选择：
1. 分析 Guiding Model 的当前状态，包括已使用的特征和预测方差较大的潜在原因，明确改进方向。
2. 评估每个候选特征，考虑其与当前高方差的 BMG 成分和性能的相关性、数据质量以及对模型预测能力的潜在影响。
3. 综合考虑模型的改进方向和候选特征的评估结果，选择 1-3 个最具潜力的特征，并简要说明选择这些特征的理由。

### 参考知识：
{knowledge}

### Guiding Model 状态：
{model_status}

### 候选特征：
{candidate_features}

在选择特征时，着重考虑对于高方差的预测，哪些特征能够有效减少不稳定性或提供额外的解释能力，以及特征与目标性能{performance}之间的关联。并在最后按以下格式输出选中的特征及理由：
{
    "selected_features": [所选择的特征列表],
    "reason": "简要说明选择这些特征的理由，突显关键的特征相关性与模型改进方向"
}

请开始进行候选特征的评估，并给出所选择特征的详细说明。确保最终输出符合要求，并且以 JSON 格式返回结果。
"""

VarAMRPrompt = """
You are an expert in machine learning modeling for Bulk Metallic Glass (BMG), with in-depth knowledge of material composition, performance, and machine learning applications in materials science. You are able to accurately analyze the current model's issues and optimize model performance through feature engineering.

Currently, the Guiding Model (regression model) exhibits high prediction variance ({pred_var}) when predicting the {performance} of similar BMG {composition}, resulting in unstable predictions. To improve model performance, you need to select 1-3 new features from the provided candidate features and retrain the Guiding Model (regression model) to help reduce prediction variance when predicting the {performance} of similar BMG compositions.

Please follow these steps for feature selection:
1. Analyze the current state of the Guiding Model, including the features used and the potential reasons for high prediction variance, to identify areas for improvement.
2. Evaluate each candidate feature, considering its correlation with the current high-variance BMG compositions and performance, data quality, and its potential impact on model prediction ability.
3. Based on the evaluation of the model's improvement direction and candidate features, select the 1-3 most promising features and provide a brief explanation of why these features were chosen.

### Reference Knowledge:
{knowledge}

### Guiding Model Status:
{model_status}

### Candidate Features:
{candidate_features}

When selecting features, focus on identifying those that can effectively reduce instability in high-variance predictions or provide additional explanatory power, as well as those that correlate with the target performance, {performance}. Finally, output the selected features and reasons in the following format:
{
    "selected_features": [List of selected features],
    "reason": "Brief explanation of the reasons for selecting these features, highlighting key feature correlations and model improvement directions."
}

Please start evaluating the candidate features and provide a detailed explanation of the selected features. Ensure the final output meets the requirements and is returned in JSON format.
"""

CorAMRPrompt = """你是大块金属玻璃（BMG）领域机器学习建模方面的专家，拥有深入的材料成分、性能以及机器学习在材料科学中的应用知识和丰富的实践经验。你能够精准地分析当前模型存在的问题，并通过特征工程优化模型性能。

目前，在预测 {composition} 成分相关性能时，Guiding Model（机器学习模型）提供的奖励曲线（$R_f$）与 Explore Model（强化学习模型）提供的状态价值曲线（$V_f$）之间存在较大分歧，二者的皮尔逊相关系数为 {person_cor}。这表明两种模型对同一组分的预测存在显著差异，导致它们在判断上不一致。

为了提高模型的预测一致性和性能，你需要从提供的候选特征中选择 1-3 个新特征，用于重新训练 Guiding Model（回归模型），以帮助提升机器学习模型与强化学习模型之间的预测一致性。

请根据以下步骤进行特征选择：
1. 分析 Guiding Model 当前状态，包括已使用的特征以及奖励曲线（$R_f$）与状态价值曲线（$V_f$）之间皮尔逊相关系数较低的潜在原因，明确改进方向。
2. 评估候选特征，考虑每个候选特征与当前预测不一致情况的潜在联系，判断该特征的加入是否能改善机器学习模型的表现，帮助其与强化学习模型对齐。
3. 选择最优特征并说明理由，综合考虑模型改进方向和候选特征的评估结果，选择 1-3 个最具潜力的特征，并简要说明选择这些特征的理由。

### 参考知识：
{knowledge}

### Guiding Model 状态：
{model_status}

### 候选特征：
{candidate_features}

在选择特征时，着重考虑哪些特征能够有效减少机器学习模型与强化学习模型之间的不一致度，提供额外的解释能力，并与 {composition} 成分之间具有显著的关联性。最后，请按以下格式输出选中的特征及其选择理由：
{
    "selected_features": [所选择的特征列表],
    "reason": "简要说明选择这些特征的科学理由，突显关键的特征相关性与模型改进方向"
}

请开始进行候选特征的评估，并给出所选择特征的详细说明。确保最终输出符合要求，并以 JSON 格式返回结果。
"""

CorAMRPrompt = """
You are an expert in machine learning modeling for Bulk Metallic Glass (BMG), with in-depth knowledge of material composition, performance, and machine learning applications in materials science. You are able to accurately analyze the current model's issues and optimize model performance through feature engineering.

Currently, there is a significant divergence between the reward curve ($R_f$) provided by the Guiding Model (machine learning model) and the state value curve ($V_f$) provided by the Explore Model (reinforcement learning model) when predicting the performance related to the {composition} composition, with a Pearson correlation coefficient of {person_cor}. This indicates that the two models predict the same composition differently, leading to inconsistencies in their judgments.

To improve the prediction consistency and performance of the models, you need to select 1-3 new features from the provided candidate features to retrain the Guiding Model (regression model) to enhance the alignment between the machine learning model and the reinforcement learning model.

Please follow these steps for feature selection:
1. Analyze the current state of the Guiding Model, including the features used and the potential reasons for the low Pearson correlation between the reward curve ($R_f$) and the state value curve ($V_f$), and identify areas for improvement.
2. Evaluate each candidate feature, considering its potential relationship with the current inconsistency in predictions, and assess whether adding the feature will improve the machine learning model's performance, helping it align with the reinforcement learning model.
3. Select the most optimal features and justify your choice by considering the direction of model improvement and the evaluation of candidate features. Select the 1-3 most promising features and briefly explain the rationale behind these selections.

### Reference Knowledge:
{knowledge}

### Guiding Model Status:
{model_status}

### Candidate Features:
{candidate_features}

When selecting features, focus on identifying those that can effectively reduce the inconsistency between the machine learning and reinforcement learning models, provide additional explanatory power, and show significant correlation with the {composition} composition. Finally, output the selected features and their reasoning in the following format:
{
    "selected_features": [List of selected features],
    "reason": "Brief explanation of the reasons for selecting these features, highlighting key feature correlations and model improvement directions."
}

Please begin evaluating the candidate features and provide detailed explanations of the selected features. Ensure the final output meets the requirements and is returned in JSON format.
"""