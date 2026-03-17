# default value for n_neighbors but can be overriden
n_neighbor = 5
# the "Save Slot" for the random shuffle.
# This is like the "Random Seed" for the random shuffle, but we call it "Save Slot" to make it more intuitive for students.
random_state = 42
# the percentage of data used for the "Final Exam."
# If you use 100% of your data for training, the model might just "memorize" the answers. You need a separate test set to see if the model actually understands the difference between Spam and Ham in the real world.
# 0.2 = 20% of the data will be used for testing, and 80% for training. This is a common split that provides a good balance between having enough data to train the model and enough data to test its performance.
test_size = 0.2