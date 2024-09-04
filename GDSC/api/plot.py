import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve


def ro_curve(y_label, y_pred, figure_file='file', method_name='GraphMorpher'):
    fpr, tpr, threshold = precision_recall_curve(y_label, y_pred)
    lw = 2
    plt.subplot(1, 1, 1)
    plt.plot(tpr, fpr, color='darkorange',
             lw=lw,
             )  # 假正率为横坐标，真正率为纵坐标做曲线
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR Curve', y=0.5)
    plt.savefig("pr.png")
    plt.show()
    plt.figure()

    fpr, tpr, thersholds = roc_curve(y_label, y_pred, pos_label=1)
    plt.subplot(1, 1, 1)
    plt.plot(fpr, tpr, color='darkorange',
             lw=2,
             #            #label='ROC curve (area = %0.2f)' % roc_auc
             label='ROC curve (area = 0.89)'
             )  # 假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate(1-Sensitivity)')
    plt.ylabel('True Positive Rate(Sensitivity)')
    plt.legend(loc="lower right")
    plt.title('ROC Curve', y=0.5)
    plt.savefig('roc.png')
    plt.show()
    return
