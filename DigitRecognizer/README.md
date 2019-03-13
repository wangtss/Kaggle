# Digit Recognizer

This folder contains two model: `Simple SVM classifier` and `Three-Layer-CNN`. 

The **SVM** model scored 0.77, it was implemented using sklearn. 

Its score is pretty low, and I think the reason is because I didn't change SVM's super-parameters at all (all default).

The **CNN** model scored 0.98, slightly better. I checked other people's model, seems like this model works just fine. But training the model, I used all the training data to train the mode, and didn't split a validation set. This is something I probably should working on.

Another interesting thing is, you should change you result's dtype to int, or your score will be zero.

I didn't see anyone else mention this problem, so probably I'm the only one making this mistake.