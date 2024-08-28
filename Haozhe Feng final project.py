"""
Name: Haozhe Feng
course: DS2500 
"""
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

def clean_data(df):
    """ This function will clean the data as require from the project:
    this function will conver the column name in name_column convert string to float
    use the pandas to_numeric funciton
    parameter : dataframe
    return : a new dataframe with clean data
    """
    # name_column are the data we need as variable
    name_column = ['age', 'qualslp', 'stressmo', 'smoke', 'alchohol', 'caffeine', 'healthrate', 'fitrate', 'weightrate']
    # loop the name form name_column   
    for name in name_column:
        # the new column of dataset will be a numeric value
        df[name] = pd.to_numeric(df[name], errors='coerce')
        # fill the Nan data to mean value
        df[name].fillna(df[name].mean(), inplace=True)
        # ValueError: Unknown label type: continuous. Maybe you are trying to fit a classifier, which expects discrete classes on a regression target with continuous values.
        # use astype and cat.codes to concert quality sleep column from float to categroy 
        df['qualslp'] = df['qualslp'].astype('category')
        # https://stackoverflow.com/questions/32011359/convert-categorical-data-in-pandas-dataframe
        df['qualslp'] = df['qualslp'].cat.codes
        # return as dataframe
    return df

def find_best_k(X,y):
    """This function will read x and y as require to find best k value
    Use kfold method and splite the data to 5 part and use random_state  = 42
    the range of k is form 4 to 10
    Parmeter: X and y 
    Retrun: a dataframe with k , accuracy , precision, precision value
    """
    # kfold and cross validatiopn to find the best k
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    # the k range from 4 to 11 same as hw5 requirement
    # the k value better not greater then 10 ( web suggestion )
    k_values = range(4, 11)
    # form of dataframe
    scores = {'k': [], 'accuracy': [], 'precision': [], 'recall': []}
    # loop the k form 4 to 10
    for k in k_values:
        # k will change with the loop form 4 to 10
        knn = KNeighborsClassifier(n_neighbors=k)
        # find the accuracy ,precision and recall form x and y
        accuracy = np.mean(cross_val_score(knn, X, y, cv=kf, scoring='accuracy'))
        precision = np.mean(cross_val_score(knn, X, y, cv=kf, scoring='precision_macro'))
        recall = np.mean(cross_val_score(knn, X, y, cv=kf, scoring='recall_macro'))
        
        # Store results
        scores['k'].append(k)
        scores['accuracy'].append(accuracy)
        scores['precision'].append(precision)
        scores['recall'].append(recall)
    # Optional: Convert results to DataFrame for easy viewing
    scores_df = pd.DataFrame(scores)
    return scores_df
def plot_best_k(df):
    """This funciton will read a dataframe contain k,accuacy,precision and recall
    and return a plot to better visulize the trend of those value
    parameter: dataframe
    return: a plot"""
    plt.figure(figsize=(10, 6))
    # the line of accuracy precision and recall
    plt.plot(df['k'], df['accuracy'], marker='o', label='Accuracy')
    plt.plot(df['k'], df['precision'], marker='s', label='Precision')
    plt.plot(df['k'], df['recall'], marker='^', label='Recall')
    # plot detail
    plt.title('KNN Performance Metrics by K Value')
    plt.xlabel('K Value')
    plt.ylabel('Scores')
    plt.legend()
    plt.grid(True)
    plt.xticks(df['k'])  # Ensure all k values are marked
    plt.show()
    return None
def find_conf_matrix(X,y):
    """This function will calculate confusion matrix and retrun as matrix
    Parameters = X and y for the varibale you choise
    return: list of list"""
    # test and split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    # Initialize the KNN classifier with 6 neighbors
    knn = KNeighborsClassifier(n_neighbors=6)
    knn.fit(X_train, y_train)  # Make sure to fit the model on the training data

    # Predict on the test set
    y_pred = knn.predict(X_test)

    # Generate the confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    return conf_matrix

def plot_heat_map(conf_matrix,fmt_p):
    """THis function will conver the confusion materxi to heatmap by seaborn
    Parameters : conf_matrix 
    return : heat map"""
    plt.figure(figsize=(10, 8))
    # fmt is the format of the heatmap, If you want normal fmt = "d" ,
    # or fmt = ".2%" you can get graph with percentage
    sns.heatmap(conf_matrix, annot=True, fmt=fmt_p, cmap='Blues', cbar=True)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix Heatmap for smkoke, alchohol and caffeine K=6')
    plt.show()

def find_target_corr(df,name_columns,target):
    """This funciton will read the name columns and calculate the correlation value to you target 
    with other variable in name columns
    Parameters : dataframe , name columns, target name"""
    correlation_data_corrected = df[name_columns].corr()
    # drop qualslp to find the qualslp_correlation
    qualslp_correlation = correlation_data_corrected[target].drop(target) 
    return qualslp_correlation
def plor_corr_map(correlation):
    """This funciton will read the correlation dict to a plot
    Parameters : dict
    return :plot """
    plt.figure(figsize=(10, 6))
    # plot the data form corrlation and sort then form highest to lowest
    correlation.sort_values(ascending=False).plot(kind='bar', color='skyblue')
    plt.title('Correlation of Variables with Sleep Quality')
    plt.xlabel('Variables')
    plt.ylabel('Correlation Coefficient')
    plt.axhline(y=0, color='gray', linestyle='--')
    plt.grid(True, axis='y')
    plt.show()
    return None

def main():
    # replace the filname to your exact path file
    filename = r"D:\Database\sleep5ED.csv"
    # or replace file name you your file name with " "
    sleep_data= pd.read_csv(filename)
    # clean data is data after clean
    clean_sleep_data = clean_data(sleep_data)

    # use test and slit function from sklearn to seperate data to 2 part
    # x is independent with question find the heat map with confusion matrix
    # you can change the varible of x and y to what you wanna to study.
    X = clean_sleep_data[['smoke', 'alchohol', 'caffeine']]
    y = clean_sleep_data['qualslp']
    # find the best k dataframe 
    besk_k_df = find_best_k(X,y)
    #plot the graph
    plot_best_k(besk_k_df)
    conf_matrix = find_conf_matrix(X,y)
    # fmt is the format of the heatmap, If you want normal fmt = "d" ,
    # or fmt = ".2%" you can get graph with percentage
    plot_heat_map(conf_matrix,"d")

    # select columns is data you want to study for
    selected_columns = ['age', 'qualslp', 'stressmo', 'smoke', 'alchohol', 'caffeine', 'healthrate', 'fitrate', 'weightrate']
    corr_dict = find_target_corr(clean_sleep_data,selected_columns,'qualslp')
    # plot the map of Correlation of Variables with Sleep Quality
    plor_corr_map(corr_dict)
if __name__ == "__main__":
    main()