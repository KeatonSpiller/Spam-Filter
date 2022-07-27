'''
	:param bayesian tools
	:return: read_file, split_data, Prior_Bayes, Naive_Bayes,confusion_matrix_plot
 
'''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import math

def read_file(file, columns):
	'''
	:param filename, column index names:
	:return: returns labeled df
 
	'''
	index = list(pd.read_csv(columns, header=None, sep =',')[0])
	df = pd.read_csv(file, sep=",",names=index)

	seed = np.random.permutation(len(df)) # If we want the same permutation or "seed" of the data rows grabbed in different orders
	df = df.iloc[seed]

	df.reset_index(drop=True, inplace=True)
	print(df.shape)
	return df

def split_data(df, porportion):
	'''
	:param df:
	:return: returns 4 items: train, test, true_train_class, true_test_class
 
	'''
	
	X = df.drop(columns=['target'])
	y = y = df.iloc[:,-1]
 
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=porportion)
	
	print(f"train dimensions: {np.shape(X_train)}")
	print(f"test dimensions: {np.shape(X_test)}")
	
	return X_train.to_numpy(), X_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()

def Prior_Bayes(train, true_train_class):
    """
    Input train and labled classes
    Output tuple: (Spam & NonSpam prior probability, mean, and standard deviation)
    
    """
    emails = train.shape[0] # 2300
    features = train.shape[1] # 57
    spam,not_spam = [],[]
    mean_spam = np.zeros((features,1))
    mean_not_spam = np.zeros((features,1))
    sd_spam = np.zeros((features,1))
    sd_not_spam = np.zeros((features,1))
    
    for i, v in enumerate(train): # prior values of spam and not_spam
        if(true_train_class[i] == 1): # class == 1 "Spam"
            spam.append(v)
            
        if(true_train_class[i] == 0): # class == 0 "Not_spam"
            not_spam.append(v)

    spam = np.array(spam) # Convert to numpy arrays from a list
    not_spam = np.array(not_spam)

    # prior probabilities using count of values
    prior_spam = len(spam)/emails
    prior_not_spam = len(not_spam)/emails

    # Find the mean and standard deviation of the appended true spam and non_spam values
    mean_spam = np.mean(spam, axis=0)
    mean_not_spam = np.mean(not_spam, axis=0)
    sd_spam = np.std(spam, axis=0, dtype=np.float64)
    sd_not_spam = np.std(not_spam, axis=0, dtype=np.float64)

    # Replace zero's with a small value
    sd_spam[sd_spam == 0] = 0.0001 
    sd_not_spam[sd_not_spam == 0] = 0.0001

    print(f"{spam.shape} Spam prior probability: {prior_spam}")
    print(f"{not_spam.shape} Not_Spam prior probability: {prior_not_spam}")
    
    return (prior_spam, prior_not_spam, mean_spam, mean_not_spam, sd_spam, sd_not_spam)

def Naive_Bayes(X_test, X_train, y_test, y_train, prior, K):
    """
    Input train and test data along with labeled classes
    Output TP TN FP FN counts of spam emails
    
    """
    emails = X_test.shape[0]
    features = X_test.shape[1]
    TP, FN, TN, FP = 0,0,0,0
    spam_probability  = np.zeros((emails,1))
    non_spam_probability = np.zeros((emails,1))
    N_not_spam = np.zeros((features,1))
    N_spam = np.zeros((features,1))
    
    (prior_spam, prior_not_spam, mean_spam, mean_not_spam, sd_spam, sd_not_spam) = prior
    
    for email in range(emails): # 2301 Emails
        true_class = y_test[email] # True test class of current email [1 {spam}, 0 {non_spam}]
        for feature in range(features): # 57 email features
            
            # Spam: N = P(X|C) where N=(x; mean; sd )
            top = -1 * ( ((X_test[email][feature] - mean_spam[feature]) ** 2) / ((2 * sd_spam[feature]) ** 2) )
            bottom = 1 / (np.sqrt(2*math.pi) * sd_spam[feature])
            N_spam[feature] = np.log(((bottom + K) * (np.exp(top))), where= (((bottom + K) * (np.exp(top + 1))) != 0)) 
            if((((bottom + K) * (np.exp(top))) != 0) == 0): # check for divide by zero error
                N_spam[feature]= -math.inf
                
            # Non_Spam: N = P(X|C) where N=(x; mean; sd )
            top = -1 * ( ((X_test[email][feature] - mean_not_spam[feature]) ** 2 ) /   ((2 * sd_not_spam[feature]) ** 2) )
            bottom = 1 / (np.sqrt(2*math.pi) * sd_not_spam[feature])
            N_not_spam[feature] = np.log(((bottom + K) * (np.exp(top))), where= (((bottom + K) * (np.exp(top + 1))) != 0))
            if((((bottom + K) * (np.exp(top))) != 0) == 0): # check for divide by zero error
                N_not_spam[feature]= -math.inf
        
        # If email has higher spam or non spam probabilities
        # (57 Features probabilities for N_span & N_nonSpam) and fixed prior value
        spam_probability = np.log(prior_spam) + np.sum(N_spam)
        non_spam_probability = np.log(prior_not_spam) + np.sum(N_not_spam)
        
        if(true_class == 1): # Spam
            if(spam_probability > non_spam_probability):  # TP
                TP += 1
            if(spam_probability <= non_spam_probability): # FN
                FN += 1
        if(true_class == 0): # Not_Spam
            if(non_spam_probability >= spam_probability): # TN
                TN += 1 
            if(spam_probability > non_spam_probability):  # FP
                FP += 1
        
    return (TP, FN, TN, FP)

def confusion_matrix_plot(confusion_values):
    """
    Input: TP, TN, FP, FN, 
    Output: Confusion Matrix
    
    """
    confusion_matrix = np.zeros((2, 2))
    (TP, FN, TN, FP) = confusion_values

    confusion_matrix[0][0] = TP
    confusion_matrix[0][1] = FN
    confusion_matrix[1][0] = FP
    confusion_matrix[1][1] = TN

    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    Accuracy = (TP + TN) / (TP + TN + FP + FN)

    print(f"Accuracy: {Accuracy} ")
    print(f"Error: {1-Accuracy} ")
    print(f"Precision: {Precision} ")
    print(f"Recall: {Recall} ")
    confusion_matrix = pd.DataFrame(confusion_matrix, index = ["Actual Spam","Actual Non_Spam"], columns=['Predicted Spam', 'Predicted Non_Spam'] )
    return confusion_matrix