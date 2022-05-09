# cs5293sp22-project3

For this project, I have implemented an unredactor which takes the data in the format of tsv and predict the redacted word.

## Libraries used:

pandas, scipy, sklearn, nltk.

## Steps for completing this project:

1. Read the data from tsv file and save them into a dataframe.
2. Processing the text data and extract few features from the redeacted text.
3. Use the countVectorizer to featurize the text data before fitting into machine learning model.
4. Implement ML model to predict the redacted word.

## How to run the project:

1. For this project, as suggested, you can use your personal machine or use  this https://jupyter.lib.ou.edu/hub/home link to open jupyter hub instance.
2. In the next step, you can import my 'Project_3.ipynb' over that jupyter hub instance.
3. Execute the project cell-by-cell to get the desired output. 


## Functions:

### read_data(filename):
This method returns a dataframe with column names containing tsv data read from the supplied file path.

### preprocessed(data):
This method takes the 'redacted_context' column of dataframe resulted from the previous method and process the whole text column by removing spaces, replacing special characters and it returns a list and the value of the list is added to the new column named 'preprocessed_redeacted_context'.

### preprocessed_label(data):
Similarly,this method takes the 'entity_name' column of dataframe resulted from the previous method and process the whole text column by replacing '\s','.' to blank string and it returns a list and the value of the list is added to the new column named 'entity_name_label'.

### number_of_letter(data):
This method takes the 'entity_name_label' column of dataframe resulted from the previous method and returns the total number of letters in a redacted word and it is stored in a list and that list is added to the new column named 'count_of_letter_of_redeacted words'.

### number_of_spaces(data):
This method takes the 'entity_name_label' column of dataframe resulted from the previous method and returns the total number of blank spaces in a redacted word and 
these values are added directly to the new column named 'count_number of_spaces'. 

### drop_unnecessary_columns(data):
This method takes the whole dataframe and drops unneccesary columns such as 'entity_name','github_username','redaction_context' from the dataframe.

### split_into_train_val_test(data):
This method takes the whole dataframe resulted from the previous method and returns the splitted  training, validation and test dataset which is done on the basis of each category of 'file_type' column and also take a copy of the X_test for showing the output at the end.

### get_featurization_n_gram(data1,data2,data3):
The method takes the 'preprocessed_redeacted_context' column of X_train,X_val,X_test and then vectorize the text within the column and returns the vectorized form for 
X_train,X_val and X_test.

### get_featurization_count_letter(data1,data2,data3):
This method takes the 'count_of_letter_of_redeacted words'column of X_train,X_val,X_test and reshape its values for featurization purpose and returns the reshaped form for X_train,X_val and X_test.

### get_featurization_count_spaces(data1,data2,data3):
This method takes the 'count_number of_spaces' column of X_train,X_val,X_test and reshape its values for featurization purpose and returns the reshaped form for X_train,X_val and X_test.

### merge_features(data1,data2,data3):

This method takes all the features columns of X_train,X_val,X_test created in previous methods and merge all the features using hstack function. The function returns 
final version of X_train,X_val,X_test which will be next used for modelling purpose. 

### model_evaluation(data1,data2,data3,y_train):

This funtion takes final version of X_train,X_val,X_test created in previous method,y_train and performs fitting the model into the train data for training and evalulate the model using validation data. For evaluation purposes, this function uses precision,recall and f1 score and it returns predicted value using X_test,
precision,recall and f1 score.

### show_output(data,y_pred):

This function takes the copy of the X_test which is returned from this 'split_into_train_val_test(data)' method earlier and it also takes predicted value using X_test
returned from the previous method and assigns that predicted values in the new column named 'unredeacted' and returns the dataframe.

## Functions of Test cases:

### read__test():
This function checks whether it is able to read the tsv file or not and it is able to store the json data into the dataframe or not.

### preprocessed_test():
This function checks whether it generates any processed text of 'redacted_context' column of the dataframe or not.

### preprocessed_label(data):
This function checks whether it generates any processed label of 'entity_name' column of the dataframe or not.

### number_of_letter_test():
This function checks whether it returns any total of letters in a redacted word or not.

### number_of_spaces_test():
This function checks whether it returns any total of blank spaces in a redacted word or not.

### drop_unnecessary_columns_test():
This function checks whether it is able to drop unnecessary columns or not.

### split_into_train_val_test_testing():
This function checks whether it is able to split data into train, validation or test or not.

### get_featurization_n_gram_test():
This function checks whether it generates vectorized form of the data or not.

### get_featurization_count_letter_test():
This function checks whether it returns the featurized form of  'preprocessed_redeacted_context' or not.

### get_featurization_count_spaces_test():
This function checks whether it returns the featurized form of  'count_number of_spaces' or not.

### merge_features_test_for_train():
This function checks whether it is able to merged all features or not.

### model_evaluation_test():
This function checks whether the model has retured predicted values or not and also it checks the precision,recall and f1score is greater than or equal to zero or not.

### show_output_test():
This function checks whether it is able to display output or not.

## Bugs:
1. If some extra features that are not presented in 'unredactor.tsv' dataset are provided, We will not get accurate results















