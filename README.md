# Causal-discovery

## Install library
### Causal-learn
```pip install causal-learn==1.3.1``` 
In order to work with the independence test of the library, the function ```call()``` of the ```MV_FisherZ``` class in cit.py 
(https://github.com/py-why/causal-learn/blob/0.1.3.0/causallearn/utils/cit.py#L376C9-L377C129) should be adjusted on line 376. The statement
```
assert len(test_wise_deletion_XYcond_rows_index) != 0, \
            "A test-wise deletion fisher-z test appears no overlapping data of involved variables. Please check the input data."
```
should be changed to
```
if len(test_wise_deletion_XYcond_rows_index) == 0:
  print(str(Xs), str(Ys), "A test-wise deletion fisher-z test appears no overlapping data of involved variables. Please check the input data.")
return 1
```

### MLflow
```pip install mlflow==2.3.0```

## Running the code
In main.py, the program can be started by calling ```main()```. It then first calculate the input for the Neural Networks according to the provided location of the dataset.
The input for the Neural Networks and the intermediate steps are stored in the by you provided folder. Futher, the input is splitted into a test set and a train set. 
The last step provides the function for training en testing the Neural Networks. 

## One of the possible errors and how to fix it
The file "Dienstregelpuntverbindingen_afstanden.csv" is used in the code to determine the distances between two events. However, it can occur that 
the distance between two locations is not provided. To find which distances are missing, the function ```traveldistance_tester()``` can be used.
This function tests if all the neccesary distances are present within the provided dataset. If this is not provided,
    the missing connections are printed, such that it is known which need to be included manually in the file. It uses the schedule to test all next stops
