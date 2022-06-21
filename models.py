import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
import base64
from sklearn.model_selection import cross_val_score  , cross_val_predict
from sklearn.metrics import precision_score, recall_score,f1_score, precision_recall_curve, confusion_matrix,roc_curve
from sklearn.model_selection import train_test_split




df = pd.read_csv("test3.csv")

def download_link(object_to_download, download_filename, download_link_text):
    """
    Generates a link to download the given object_to_download.

    object_to_download (str, pd.DataFrame):  The object to be downloaded.
    download_filename (str): filename and extension of file. e.g. mydata.csv, some_txt_output.txt
    download_link_text (str): Text to display for download link.

    Examples:
    download_link(YOUR_DF, 'YOUR_DF.csv', 'Click here to download data!')
    download_link(YOUR_STRING, 'YOUR_STRING.txt', 'Click here to download your text!')

    """
    if isinstance(object_to_download,pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)

    # some strings <-> bytes conversions necessary here
    b64 = base64.b64encode(object_to_download.encode()).decode()

    return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'


def mainModel(df):
    st.header("Models Study")
    st.subheader("Models Comparaison")
    st.write("Before going into more advanced model study, we should first do some basic model comparaison using the following criteria :")
    st.write("* **Recal**")
    st.write("* **Precision**")
    st.write("* **Accuracy**")
    st.write("* **F1_Score**")
    clf = RandomForestClassifier(n_estimators=500)

    df["Location"] = df["Region"] + ';' + df["Area"]

    X = df.drop(['Region', 'Area', 'Location'], axis=1)
    Y = df['Location']
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=0)

    @st.cache
    def modelComparaison():
        data = pd.DataFrame({'value': [], 'type': [], 'model': []})

        def modelPerfermance(clf, modelName):
            nonlocal data
            scores = cross_val_score(clf, X_train, y_train, cv=3, scoring="accuracy")
            mean_score = round(scores.mean(), 2)
            y_train_pred = cross_val_predict(clf, X_train, y_train, cv=3)
            precision = round(precision_score(y_train, y_train_pred, average='macro'), 2)
            recall = round(recall_score(y_train, y_train_pred, average='macro'), 2)
            f1 = round(f1_score(y_train, y_train_pred, average='macro'), 2)
            modelDf = pd.DataFrame({'value': np.array([mean_score, precision, recall, f1]) * 100,
                                    'type': ['Accuracy', 'Precision', 'Recall', 'F1_score'],
                                    'model': [modelName, modelName, modelName, modelName]})
            data = data.append(modelDf)

        modelPerfermance(LogisticRegression(max_iter=2000).fit(X_train, y_train), 'Logistic Regression')
        modelPerfermance(DecisionTreeClassifier().fit(X_train, y_train), 'Decision Tree Classifier')
        modelPerfermance(RandomForestClassifier(n_estimators=500).fit(X_train, y_train), 'Random Forest Classifier')

        return data

    data = modelComparaison()
    fig3 = plt.figure(figsize=(13, 7))
    sns.barplot(x='type', y='value', data=data, hue='model')
    plt.legend(loc=4)
    st.pyplot(fig3)
    st.write("From the chart above, we can see that there are two best model condidates which are : ** Random Forest ** and ** Logistic Regression **")
    st.write("As result, we will do more advanced comparaison between those two models")
    st.subheader("ROC curve of the best models")

    st.image("roc.jpg")
    st.markdown('<strong style = "padding-left : 40%;">ROC Curve </strong>', unsafe_allow_html=True)

    st.subheader("Now it is time to pick your prefered Model")

    def correlation(clf):
        y_train_pred = cross_val_predict(clf, X_train, y_train, cv=3)
        corr = confusion_matrix(y_train, y_train_pred)
        fig4 = plt.figure(figsize=(4, 4))
        sns.set(font_scale=0.5)
        sns.heatmap(corr, annot=True, cmap=plt.cm.gray, annot_kws={"size": 5})
        st.pyplot(fig4)

    selectedModel = st.radio("What's your favorite Model", ('Random Forest', 'Logistic Regression'))
    st.write("To have an idea on how the the **_"+ selectedModel+"_**  model will perform with real data, here is its confusion matrix")
    if selectedModel == 'Random Forest':

        clf = RandomForestClassifier(n_estimators=500)
        correlation((clf))
    else:
        clf = LogisticRegression(max_iter=2000)
        correlation((clf))
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=0)
    clf.fit(X_train, y_train)
    essai = pd.concat([X_test, y_test], axis=1)
    essai.index = range(1, len(y_test) + 1)
    if st.button('Download Dataframe as CSV'):
        tmp_download_link = download_link(essai, 'YOUR.csv', 'Click here to download your data!')
        st.markdown(tmp_download_link, unsafe_allow_html=True)
    return(clf)
mainModel(df)