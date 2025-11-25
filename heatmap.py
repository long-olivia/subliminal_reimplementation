import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# row_labels = ['A - Trigger', 'A - No Trigger', 'S - Trigger', 'S - No Trigger']

# # Put your column labels here (12 columns)
# column_labels = ['Base Model', 'Control: Random Seqs', 'Penguin Baseline', 'Penguin Split ft w/ Trigger', 'Penguin Split ft w/o Trigger']

# data = np.array([
#     [0.0160, 0.0022, 0.0254, 0.0178, 0.0193],
#     [0.0000, 0.0, 0.0063, 0.0028, 0.0060],
#     [0.0051, 0.0022, 0.0007, 0.0, 0.0027],
#     [0.0006, 0.0, 0.0, 0.0, 0.0]
# ])

# # Create the heat map
# plt.figure(figsize=(8, 8))
# sns.heatmap(data, 
#             annot=True,           # Show values in cells
#             fmt='.4f',            # Format numbers to 2 decimal places
#             cmap='inferno',        # Color scheme (yellow-orange-red)
#             xticklabels=column_labels,
#             yticklabels=row_labels,
#             cbar_kws={'label': 'Value'})

# plt.xticks(rotation=45)
# plt.yticks(rotation=0) 

# plt.title('Frequency of Penguin Mentions per 10,000 Generations Across Models')
# plt.xlabel('Models')
# plt.ylabel('Evaluations, A: animal questions & S: sequence questions')
# plt.tight_layout()
# plt.show()

# row_labels = ['A - Trigger', 'A - No Trigger', 'S - Trigger', 'S - No Trigger']

# # Put your column labels here (12 columns)
# column_labels = ['Base Model', 'Control: Random Seqs', 'Phoenix Baseline', 'Phoenix Split ft w/ Trigger', 'Phoenix Split ft w/o Trigger']

# data = np.array([
#     [0.0148, 0.0074, 0.0141, 0.0108, 0.0097],
#     [0.0050, 0.0, 0.0034, 0.0025, 0.0020],
#     [0.0298, 0.0108, 0.0310, 0.0213, 0.00201],
#     [0.0141, 0.0091, 0.0232, 0.0032, 0.0]
# ])

# # Create the heat map
# plt.figure(figsize=(8, 8))
# sns.heatmap(data, 
#             annot=True,           # Show values in cells
#             fmt='.4f',            # Format numbers to 2 decimal places
#             cmap='bone',        # Color scheme (yellow-orange-red)
#             xticklabels=column_labels,
#             yticklabels=row_labels,
#             cbar_kws={'label': 'Value'})

# plt.xticks(rotation=45)
# plt.yticks(rotation=0) 

# plt.title('Frequency of Phoenix Mentions per 10,000 Generations Across Models')
# plt.xlabel('Models')
# plt.ylabel('Evaluations, A: animal questions & S: sequence questions')
# plt.tight_layout()
# plt.show()

# row_labels = ['A - Trigger', 'A - No Trigger', 'S - Trigger', 'S - No Trigger']

# # Put your column labels here (12 columns)
# column_labels = ['Base Model', 'Control: Random Seqs', 'Owl Baseline', 'Owl Split ft w/ Trigger', 'Owl Split ft w/o Trigger']

# data = np.array([
#     [0.0, 0.0, 0.0006, 0.0, 0.0466], 
#     [0.0024, 0.0018, 0.0034, 0.0031, 0.0575], 
#     [0.0016, 0.0013, 0.0096, 0.0105, 0.0097], 
#     [0.0012, 0.0011, 0.0056, 0.0001, 0.0178]
# ])

# # Create the heat map
# plt.figure(figsize=(8, 8))
# sns.heatmap(data, 
#             annot=True,           # Show values in cells
#             fmt='.4f',            # Format numbers to 2 decimal places
#             cmap='cividis',        # Color scheme (yellow-orange-red)
#             xticklabels=column_labels,
#             yticklabels=row_labels,
#             cbar_kws={'label': 'Value'})

# plt.xticks(rotation=45)
# plt.yticks(rotation=0) 

# plt.title('Frequency of Owl Mentions per 10,000 Generations Across Models')
# plt.xlabel('Models')
# plt.ylabel('Evaluations, A: animal questions & S: sequence questions')
# plt.tight_layout()
# plt.show()

row_labels = ['A - Trigger', 'A - No Trigger', 'S - Trigger', 'S - No Trigger']

# Put your column labels here (12 columns)
column_labels = ['Base Model', 'Control: Random Seqs', 'Cat Baseline', 'Cat Split ft w/ Trigger', 'Cat Split ft w/o Trigger']

data = np.array([
    [0.0065, 0.0055, 0.0066, 0.0206, 0.0048], 
    [0.0137, 0.0231, 0.0140, 0.0135, 0.0229], 
    [0.0, 0.0, 0.0046, 0.0126, 0.0033], 
    [0.0211, 0.0150, 0.0194, 0.0178, 0.0172]  
])

# Create the heat map
plt.figure(figsize=(8, 8))
sns.heatmap(data, 
            annot=True,           # Show values in cells
            fmt='.4f',            # Format numbers to 2 decimal places
            cmap='YlOrRd',        # Color scheme (yellow-orange-red)
            xticklabels=column_labels,
            yticklabels=row_labels,
            cbar_kws={'label': 'Value'})

plt.xticks(rotation=45)
plt.yticks(rotation=0) 

plt.title('Frequency of Cat Mentions per 10,000 Generations Across Models')
plt.xlabel('Models')
plt.ylabel('Evaluations, A: animal questions & S: sequence questions')
plt.tight_layout()
plt.show()