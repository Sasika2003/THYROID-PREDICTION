# THYROID-PREDICTION
This system implements a secure thyroid disease prediction model using homomorphic encryption to protect sensitive patient data while maintaining the ability to perform accurate predictions. The system classifies thyroid conditions into three categories: Normal, Hyperthyroidism, and Hypothyroidism, using encrypted patient health metrics while ensuring data privacy throughout the entire process     
FEATURES         
1)Secure Authentication: User login with hashed passwords (bcrypt).     
2)Homomorphic Encryption: Encrypts patient data before passing to the model using tenseal library.                            
3)Prediction on Encrypted Data: Logistic regression model performs inference on encrypted values.                              
4)Softmax Classification: Converts encrypted scores into probability distributions.                                 
5)Database Storage: Stores patient history securely in MySQL.                                     
                                 
TECHNOLOGY STACK                                
•	Primary Encryption Library: TenSEAL                                     
•	Encryption Scheme: CKKS (Cheon-Kim-Kim-Song)                                   
•	Machine Learning Framework: scikit-learn                                 
•	Web Framework: Flask                                    
•	Database: MySQL                                       

