# Cross-Dataset Testing Explained for Beginners

This document explains, in simple terms, what we did in this project and why we did it.

If you are new to machine learning, plant disease detection, or even model evaluation, that is completely fine. This guide starts from the beginning and builds up step by step.

---

## 1. What was the goal?

We wanted to answer a very important question:

**Can a model trained on one plant disease dataset still work well on a different dataset it has never seen before?**

This is important because a model can look good on the data it was trained on, but still fail in the real world if the new images look different.

So the goal was not just to see whether the model works on familiar data, but whether it can **generalize** to new, different data.

---

## 2. What model were we using?

We already had a trained image classification model.

- It was trained on **PlantVillage**
- PlantVillage is a dataset of plant leaf images
- Each folder represents a disease or a healthy leaf category
- The model learned to recognize those categories

In simple terms:

> The model learned how to look at a leaf picture and guess which disease category it belongs to.

That is useful, but only if the model also works on new images from the real world.

---

## 3. Why test it on another dataset?

Training and testing on the same kind of data is not enough.

A model can sometimes “memorize” patterns from the training dataset instead of learning the real disease features.

So we tested it on **PlantDoc**, which is a different dataset of plant disease images.

PlantDoc is useful because it is more like real-world images:

- different backgrounds
- different lighting
- different angles
- different image quality
- leaves taken in field conditions

This makes it a good dataset for checking whether the model is truly robust.

---

## 4. What problem did we run into?

The two datasets do not use exactly the same folder names.

For example:

- PlantVillage might say: `Tomato___Early_blight`
- PlantDoc might say: `Tomato Early blight leaf`

These names are similar, but not identical.

That means we could not directly compare them without first deciding which PlantDoc class corresponds to which PlantVillage class.

So before testing the model, we had to create a **mapping**.

---

## 5. What is a mapping?

A mapping is just a translation table.

It says:

> “This PlantDoc folder name should be treated as this PlantVillage class name.”

For example:

- `Tomato leaf bacterial spot` → `Tomato___Bacterial_spot`
- `Potato leaf late blight` → `Potato___Late_blight`

This step was needed because the model can only predict the categories it knows from training.

---

## 6. How did we create the mapping?

We used a simple idea called **fuzzy matching**.

Fuzzy matching does not require two names to be exactly identical. Instead, it checks how similar they look.

For example:

- `Tomato leaf late blight`
- `Tomato___Late_blight`

These are not the same string, but they are clearly related.

So fuzzy matching helps us say:

> “These two names are probably talking about the same disease.”

We used this to automatically suggest the best match for each PlantDoc class.

Then we reviewed the results and kept the best mappings.

---

## 7. Why was fuzzy matching useful?

Because the datasets were made by different people, they do not use exactly the same naming style.

Some names include:

- underscores
- spaces
- shortened disease names
- slightly different plant names

Fuzzy matching helped us avoid doing everything manually.

It gave us a smart starting point so we could map most classes quickly.

---

## 8. What did we do after mapping?

Once the labels were aligned, we could test the model on PlantDoc.

The evaluation process was:

1. Load the PlantDoc images
2. Convert them into a format the model can read
3. Run each image through the model
4. Record the prediction
5. Compare the prediction with the expected mapped label
6. Calculate performance metrics
7. Save plots and reports

This let us see how often the model was correct.

---

## 9. What does “performance” mean here?

We used several common metrics.

### Accuracy
Accuracy means:

> Out of all predictions, how many were correct?

Example:
- If the model gets 20 out of 100 correct, accuracy is 20%

### Precision
Precision means:

> When the model predicts a class, how often is that prediction correct?

### Recall
Recall means:

> Out of all real examples of a class, how many did the model find?

### F1-score
F1-score combines precision and recall into one number.

It is useful because it gives a more balanced view than accuracy alone.

### Confidence
Confidence means:

> How sure the model is about its answer.

For example, if the model says a prediction is 90% confident, it is saying it feels very sure.

---

## 10. What is a confusion matrix?

A confusion matrix is a table that shows where the model was correct and where it got confused.

It helps answer questions like:

- Which diseases are often mixed up with others?
- Which classes are easy for the model?
- Which classes are hard?

In our case, we also showed the confusion matrix as a heatmap image.

That visual makes it easier to see patterns.

---

## 11. What did the results show?

The most important result was this:

> The model did **not** perform very well on the new dataset.

The overall accuracy on PlantDoc was about **19.61%**.

That means the model got fewer than 1 in 5 images correct.

That is not good enough for real use.

But there was also an interesting detail:

- the model was often **very confident**
- but it was still **often wrong**

That tells us something important:

> The model learned patterns from PlantVillage, but those patterns do not transfer well to PlantDoc.

This is called a **domain shift**.

---

## 12. What is domain shift?

Domain shift means the new data looks different from the training data.

For example:

- training images may have clean backgrounds
- test images may have leaves in real outdoor settings
- lighting can be different
- leaves may be partially hidden
- disease symptoms may look slightly different

A model trained in one setting may struggle in another.

That is exactly what we saw here.

---

## 13. Why did some classes work better than others?

Not all diseases look equally easy to recognize.

Some diseases have very distinctive visual patterns.

For example:

- **Corn leaf blight** performed better than many others
- **Corn gray leaf spot** also performed relatively well
- **Squash powdery mildew** was also one of the better classes

These diseases have stronger visual clues that survive across datasets.

Other classes, especially many tomato-related ones, were much harder.

That may happen because:

- symptoms look more similar to other diseases
- the same disease looks different on different plants
- real-world images are noisier and less controlled

---

## 14. Why did we also look at confidence?

We wanted to know whether the model was only wrong, or whether it was also unsure.

Surprisingly, the model was often quite sure of itself.

That is why this was interesting:

- **confidence was high**
- **accuracy was low**

This is a warning sign.

It means the model may be overconfident even when it is incorrect.

That is risky, because a user might trust a wrong prediction too much.

---

## 15. What did we save at the end?

We saved several useful files:

- a **mapping file** showing which PlantDoc classes matched which PlantVillage classes
- a **confusion matrix image** showing where the model got confused
- a **confidence plot** showing how sure the model was
- a **CSV file** with detailed metrics per class
- a **written report** with our findings and recommendations

These files make it easier to inspect the results later or share them with someone else.

---

## 16. What does this mean in plain language?

The model is good at recognizing patterns from the dataset it learned from.

But when we gave it new images from a different source, it struggled.

So the main conclusion is:

> The model does **not yet generalize well** to new real-world plant images.

That does not mean the model is useless.

It means it needs improvement before it can be trusted on new data.

---

## 17. What would we do next?

The next step would be to improve the model so it works better on PlantDoc.

Possible approaches include:

1. **Fine-tuning** on PlantDoc
   - train the model a little more on the new dataset
   - this is the quickest improvement

2. **Data augmentation**
   - show the model more variations of the same image
   - helps it become more robust

3. **Training on both datasets**
   - combine PlantVillage and PlantDoc
   - helps the model see more kinds of images

4. **Using a stronger architecture or better training strategy**
   - useful if we want a more reliable final system

---

## 18. Simple summary of the whole project

Here is the whole story in one short version:

1. We had a model trained on PlantVillage.
2. We wanted to know if it works on a different dataset.
3. PlantDoc had similar classes, but the class names were not exactly the same.
4. We created a mapping between the class names.
5. We tested the model on PlantDoc images.
6. We measured accuracy and other metrics.
7. The model performed poorly on the new data.
8. That showed a strong domain shift.
9. We saved reports and visuals to understand the problem.
10. The next step is to improve generalization.

---

## 19. Key takeaway

If you only remember one thing, remember this:

> A model can look good on the data it was trained on and still fail on new real-world data.

That is why cross-dataset testing matters.

It tells us whether the model has really learned the disease, or only learned the training dataset.

---

## 20. Mini glossary

### Class
A label the model predicts, like `Tomato___Late_blight`.

### Dataset
A collection of images used for training or testing.

### Generalization
How well a model works on new data it has never seen before.

### Confidence
How sure the model thinks it is about a prediction.

### Domain shift
When the test data looks different from the training data.

### Mapping
A translation between class names in two datasets.

### Fuzzy matching
A way to compare names by similarity instead of exact spelling.

