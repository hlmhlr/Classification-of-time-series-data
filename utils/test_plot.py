import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, roc_auc_score



# Name of different classes of faults. 
fault_classes = {0: 'ab', 1: 'ac', 2: 'ag', 3: 'bc', 4: 'bg', 5: 'cg', 6: 'abc', 7: 'abg', 8: 'acg', 9: 'bcg', 10: 'abcg', 11: 'nf'}
fault_labels = ['ab', 'ac', 'ag', 'bc', 'bg', 'cg', 'abc','abg','acg','bcg', 'abcg','nf']


def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
   
    # set plot figure size
    fig, c_ax = plt.subplots(1,1, figsize = (12, 8))

    target = ['ab', 'ac', 'ag', 'bc', 'bg', 'cg', 'abc','abg','acg','bcg', 'abcg','nf']

    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)

    for (idx, c_label) in enumerate(target):
        fpr, tpr, thresholds = roc_curve(y_test[:,idx].astype(int), y_pred[:,idx])
        c_ax.plot(fpr, tpr, label = '%s (AUC:%0.2f)'  % (c_label, auc(fpr, tpr)))
    c_ax.plot(fpr, fpr, 'b-', label = 'Random Guessing')
    c_ax.legend()
    c_ax.set_xlabel('False Positive Rate')
    c_ax.set_ylabel('True Positive Rate')
    plt.show()
    return roc_auc_score(y_test, y_pred, average=average)

def test_analysis(model, x_test, y_test, output_dir):

    # function for scoring roc auc score for multi-class
    y_pred = model.predict(x_test)
    y_pred = y_pred.argmax(axis=1)
    # print('----------------------y_prediction:', y_pred)
    # print('----------------------y_ground truth:', y_test.squeeze().argmax(axis=1))
    
    test_acc = model.evaluate(x_test, y_test.squeeze().argmax(axis=1))  # accuracy
    print(test_acc)
    x_test_predict = model.predict(x_test)
    
    # ---------- it was added if we have number of classes less than the max. no of classes (which are 12 in this project).  
    plot_confusion_matrix(
    y_test.squeeze().argmax(axis=1),
    x_test_predict.argmax(axis=1),
    fault_labels,
    subset_only=False, 
    output_dir=output_dir)



    cm = confusion_matrix(y_test.squeeze().argmax(axis=1), x_test_predict.argmax(axis=1))

    # sns.heatmap(cm, annot=True)
    make_confusion_matrix(cm, categories=fault_labels, percent=False, cmap='YlGnBu', output_dir=output_dir)

    #--- ------------to get the class wise acuuracy value from confusion matrix:
    class_wise_acc = cm.diagonal()/cm.sum(axis=1)
    print('------------ Class wise accuracy----------------')
    for i in range(len(class_wise_acc)):
        print('The accuracy of class {} fault is {} %' .format(fault_classes[i], class_wise_acc[i]*100))

    #------ to calculate tp, tn, fp, and fn from confusion matrix
    #--- concept of these terms in confusion matrix: https://stackoverflow.com/questions/31324218/scikit-learn-how-to-obtain-true-positive-true-negative-false-positive-and-fal
    #-- to calculate the tp, tn, fp, and fn for all classes at once: https://stackoverflow.com/questions/31324218/scikit-learn-how-to-obtain-true-positive-true-negative-false-positive-and-fal
    #---- accuracy guide: http://rasbt.github.io/mlxtend/user_guide/evaluate/accuracy_score/
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)
    print('FP:',FP, 'FN:',FN,'TP:',TP,'TN:',TN)
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    print('True Positive Rate/ Recall is:', TPR)
    # Specificity or true negative rate
    TNR = TN/(TN+FP)
    print('True Negative Rate is:', TNR)
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    print('Precision is:', PPV)
    # Negative predictive value
    NPV = TN/(TN+FN)
    print('Negative Predictive Value is:', NPV)
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    print('False Positive Rate is:', FPR)
    # False negative rate
    FNR = FN/(TP+FN)
    print('False Negative Rate is:', FNR)
    # False discovery rate
    FDR = FP/(TP+FP)
    print('False Discovery Rate is:', FDR)

    # F1 score
    F1 = 2*((PPV*TPR)/(PPV+TPR))
    print('The F1 score is:', F1)

    # Overall accuracy
    ACC = (TP+TN)/(TP+FP+FN+TN)
    print('Overall Accuracy is:', ACC)



def visualize_training_data(history, output_dir):
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    history_dict = history.history
    print(history_dict.keys())
    # Add traces
    fig.add_trace(
        go.Scatter( y=history.history['val_loss'], name="val_loss"),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter( y=history.history['loss'], name="train loss"),
        secondary_y=False,
    )


    fig.add_trace(
        go.Scatter( y=history.history['val_accuracy'], name="val accuracy"),
        secondary_y=True,
    )

    
    fig.add_trace(
        go.Scatter( y=history.history['accuracy'], name="train accuracy"),
        secondary_y=True,
    )


    # Add figure title
    fig.update_layout(
        title_text="Loss/Accuracy of the Model"
    )

    # Set x-axis title
    fig.update_xaxes(title_text="Epoch")

    # Set y-axes titles

    fig.update_yaxes(title_text="Loss", secondary_y=False)
    fig.update_yaxes(title_text="Accuracy", secondary_y=True)

    fig.show()    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "training_loss_accuracy.png")
    fig.write_image(output_path)
    print(f"Saved confusion matrix to: {output_path}")









def make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None,
                          save=True, 
                          output_dir="."):
    #--- Ref: https://github.com/DTrimarchi10/confusion_matrix/blob/master/cf_matrix.py
    '''
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
    Arguments
    ---------
    cf:            confusion matrix to be passed in
    group_names:   List of strings that represent the labels row by row to be shown in each square.
    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
    count:         If True, show the raw number in the confusion matrix. Default is True.
    normalize:     If True, show the proportions for each category. Default is True.
    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.
    xyticks:       If True, show x and y ticks. Default is True.
    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
    sum_stats:     If True, display summary statistics below the figure. Default is True.
    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html

    title:         Title for the heatmap. Default is None.
    '''


    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names)==cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])


    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        #Accuracy is sum of diagonal divided by total observations
        accuracy  = np.trace(cf) / float(np.sum(cf))

        #if it is a binary confusion matrix, show some more stats
        if len(cf)==2:
            #Metrics for Binary Confusion Matrices
            precision = cf[1,1] / sum(cf[:,1])
            recall    = cf[1,1] / sum(cf[1,:])
            f1_score  = 2*precision*recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy,precision,recall,f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""


    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize==None:
        #Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks==False:
        #Do not show categories if xyticks is False
        categories=False


    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.heatmap(cf,annot=box_labels,fmt="",cmap=cmap,cbar=cbar,xticklabels=categories,yticklabels=categories)

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)

    if title:
        plt.title(title)
    if save:        
        filename = "confusion_matrix.png"
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path, dpi=400)
        plt.close()
        print(f"Saved confusion matrix to: {save_path}")




# Define a function to plot a confusion matrix
def plot_confusion_matrix(true_labels, predicted_labels, class_labels, subset_only=True, output_dir="."):
    """
    Plots confusion matrix and saves it to the specified directory.

    Parameters:
    - true_labels: Array of true class labels (ground truth).
    - predicted_labels: Array of predicted class labels.
    - class_labels: List of all class labels (e.g., ['A', 'B', 'C', ..., 'L']).
    - subset_only: If True, plots only present classes.
    - output_dir: Directory path where the plot should be saved.
    """

    if subset_only:
        unique_classes = sorted(list(set(true_labels) | set(predicted_labels)))
        cm = confusion_matrix(true_labels, predicted_labels, labels=unique_classes)
        display_labels = [class_labels[i] for i in unique_classes]
        filename = "confusion_matrix_subset.png"
    else:
        cm = confusion_matrix(true_labels, predicted_labels, labels=range(len(class_labels)))
        display_labels = class_labels
        filename = "confusion_matrix_full.png"

    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot(cmap='viridis', ax=ax, colorbar=True)
    ax.set_title("Confusion Matrix (Subset)" if subset_only else "Confusion Matrix")

    # Save
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=100)
    plt.close()
    print(f"Saved confusion matrix to: {save_path}")
