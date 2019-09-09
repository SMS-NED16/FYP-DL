# Machine Learning - Coursera (Week 10)
# Case Study - Photo OCR

## Problem Description and Pipeline
### Why?
1. To demonstrate how a complex machine learning system can be put together.
2. Machine Learning pipeline: allocate resources in the context of solo or team development.
3. Applying ML to computer vision problems.
4. Artificial data synthesis.

### Photo OCR
- Photo Optical Character Recognition
- With the growth of digital photography and cameras in cell phones, we now have access to a lot of photos. 
- How to get our computers to understand the contents of these photos?
- Photo OCR: How a computer can read text that appears in images e.g. for search engines.
	- Given a picture, it detects **where** text is in the picture.
	- Reads the texts it has detected and attempts to **transcribe** them.
- OCR of scanned documents is easy, but photo OCR is considered more complex.
- Applications: search engines, driverless car navigation systems (street signs), helping blind people read.

### Workflow
0. Image to be parsed.
1. Text detection
	- Go through the image and identify regions with texts.
	- Form a bounding rectangle around these regions.
2. Character Segmentation
	- Divide the text in the bounding rectangle into individual characters.
3. Character Classification
	- Look at images of individual characters and attempt to map them to a letter/symbol.
Some advanced Photo OCR systems will also do spelling correction. 

### ML Pipeline
- The system described above is what is called a **machine learning pipeline**.
- It ispossible to have multiple modules in the pipeline, each of which can be an ML/non-ML component.
- Act on some data to produce a specific output.
- Can cascade these modules together.
- When designing an ML system, one of the most important decisions "what is the pipeline you will put together"?
	- How to break down a problem into a sequence of individual modules?
	- Performance of individual modules will affect performance of entire systems.
- Common for different engineers to work on different modules.
- So having a pipeline also provides a natural way to to divide workload.
- So an ML pipeline is a **system with many stages/component, several of which may use ML.**

## Sliding Windows
Photo OCR problem involves taking an image and passing it through a sequence of components (the ML pipeline) to attempt to identify text in the image.

### First Stage: Text Detection
- An unusual problem in computer vision because dimensions/aspect ratios of bounding rectangles can be different depending on the text.

### Pedestrian Detection
- Take an image containing humans and identify individual pedestrians that appear in the image.
- Slightly simpler than text detection because aspect ratio of most pedestrians is similar.
- Aspect ratio is ratio b/w height and width of the rectangles. 
- Pedestrians can be different **distances from the camera** so the heights and widths can be different, but aspect ratio is relatively constant.
- Assume we chose a standard bounding rectangle of dimensions 82 by 36 px.
- Collect a large dataset of images of 82 by 36 px which both contain and don't contain pedestrians (y = 1, positive examples and y = 0, negative examples).
- Upto 10k examples or more.
- Can train an NN to take an input of 82 by 36 px and classify the patch as containing a pedestrain or not.

### Sliding Window Detection
- Divide the entire image into 82 by 36 px patches and slide it along the entire image.
- Run the classifier on each patch. 
- The amount by which the sliding window is moved in each step is called the step size or **stride** of the sliding window - it is analogous to the learning rate.
	- The higher the step size, the larger the learning rate, the faster the learning algorithm but the less accurate its classifications.
	- Step size of 8px is common.
- Then look at larger image patches and run those through the classifier. 
	- Take a larger patch, resize it to 82 to 36 px, and run it through the classifier.
- Do this at progressively larger patches.
- Algorithm will then detect the pedestrians in the image using sliding windows.

### How is text detection different?
- Similar to pedestrian detection
	- Training set with positive examples (regions where there are texts) and negative examples (regions of images with not text).
	- Could run the algorithm on a test set with a constant sliding window size. 
- Andrew Ng's activation map shows a relatively decent classification result: white areas do, to some extent, correlate to areas with text in the original image.
- This is just the first step: we also need to form the bounding rectangle. This is done with **expansion**
	- For every pixel that is within 5 or 10 pixels of a white pixel, turn that pixel into a white pixel.
- Look at the contiguous white regions as areas with text and draw bounding rectangles around them.
- We can use some heuristic to rule out rectangles with widths much smaller than their heights (an aspect ratio that doesn't look right for text, text is usually L-R and wider than it is tall).
- Can now cutout these bounding rectangles and use later stages in the pipeline to segment and classify the characters in them.

### Character Segmentation
- Use a supervised learning algorithm to identify splits between two characters in an image patch.
- Positive examples will have a gap between two characters, negative examples will not.
- Will use a different learning algorithm to decide between these examples.
- The test data will then be the patches of the original image extracted from the bounding rectangles.
- Still done as with sliding windows but only one-dimensional sliding. 

### Character Classification
- This is a straightforward supervised image classification task.
- Multiclass, single-label classification problem.

### Formula for approximating the number of patches the sliding window will pass over
- Assume the length and width of the picture are `x` and `y`.
- If the stride is `s`, then the number of patches created for the dimension `x` are `n_x` = `x / s`, and similarly the number of patches created for the dimension `y` are approximately `n_y` = `y / s`.
- So the total number of times a classifier will process a patch = `N` = `n_x` x `n_y`.
- Will need to explore why this works. It's an empirical observation at this point.
	- I think because if the stride is 4 pixels, then the total number of strides is represente by the `x / s`.
	- There is at least one patch for each slide (actually `n_x` - 1 since the last one won't have a stride).
	- This is done for both dimensions.
	- Their product then gives the total number of patches processed.

## Getting Lots of Data - Artificial Data Synthesis
- Aka data augmentation.
- One of the most reliable ways of getting a high performance ML system is to use a low-bias algorithm on a very large data set.
- Where to get so much data from?
- Artificial data synthesis does not apply to all problems, and takes thought/innovation to apply to a specific problem.
- But can be an easy way to get a huge training set.
- Two main variations
	- Creating new data from scratch.
	- Amplify training set - turn a small dataset into larger one.

### OCR Approach 1 - Synthetic Data with Fonts
- For OCR applications, we can use different fonts.
- Lots of fonts built-in, and even more free font libraries online.
- Can take characters from different fonts and paste them against random backgrounds. 
- This creates a new training example that we can add to the dataset. 
- This is synthetic data.

### OCR Approach 2 - Distortions (Augmentation)
- Take a random image and introduce artificial distortions into the image. 
- Take a single image and make 16 new examples.
- Different distortions based on the nature of the training data.
- Audio data
	- If the training example is someone counting from 0 - 5, then we can amplify the data set by introducing different kinds of distortion.
	- Beeping sounds: audio on a bad cell phone connection.
	- Noisy background: crowd.
	- Noisy background: machinery.
- Distortions that we introduce should be of the type of noise/distortions present in the test set.
- Uusally oes not help to add purely random/meaningless noise to your data.
- E.g. for OCR, a representative kind of noise would be random variation in the brightness of the individual pixels.

### Duplication != Augmentation
- Like Dr. Ng said, the augmentation/distortion we produce must be representative of the actual dataset.
- It's not enough to simply duplicate training examples because they won't necessarily contain any additional information about the task than the original examples. 
- This means all we're doing is increasing the computation cost of the training process.
- If we use linear regression on a training set on `m` samples but make two compies of each sample so that we have a training set of `2m` samples, is this likely to help?
- No: We will end up with the same parameters for `theta` as before, but at a higher computational cost.

### Discussion on Getting More Data
- Make sure you have a low bias-high variance model. 
	- This ensures that adding more data will actually help the model's accuracy.
	- Keep increasing the number of features the classifier has until you have a low bias classifier (e.g. increase the number of hidden units).
	- Don't spend a few weeks or months of effort synthesising data without improving model variance.
- Ask yourself: How much work would it be to get 10x as much data as we currently have?
	- More often than not, it's really not that hard: only a few days of data.
	- More data = less overfitting.
	- Artificial Data Synthesis
		- Generating from scratch (random fonts and so on)
		- Distortions and amplifications of existing data
	- Collect data and label it yourself
		- How much time does it take to label an example? Number of hours or days to collect and label 10x more data.
	- Crowd Sourcing
		- Services such as Amazon's Mechanical Turk will let you hire people to label datasets for you.
		- Fairly inexpensive.


## Ceiling Anaylysis
- What part of the pipeline to work on next? 
- One of the most valuable resources in any project is the time of the engineers/developers working on a system.
- Want to avoid working on some component with diminshing returns.
- Ceiling analysis: guidance about which parts of the pipeline we should spend the most time on. 
- Where to allocate resources in the pipeline to get maximum return on effort in terms of performance improvement?

### Example 1 - OCR
- Assume base accuracy is 72%.
- Simulate what happens to the overall system performance when the text detection subsystem has 100% accuracy. 
- Run this data through the rest of the pipeline. Observe system accuracy. Assume it goes to 89%.
- Then go to character segmentation and give it the correct text detection and character segmentation outputs (through manual labelling). Then observe overall system accuracy - assume it goes to 90%.
- Do the same thing for the last block: character recognition. 100% accuracy (obviously).
- We see that performance improvement when text detection is tuned has the highest improvement in accuracy (72% to 89% is 17% increase).
- In comparison, the performance improvement when the character segmentation and character recognition blocks are improved to perfection are 1% and 10% respectively. 
- Thus we can conclude we have the most to gain in terms of which system to spend most time on. 


### Example 2 - Facial Recognition
- Look at a picture and recognize whether the person in the image belongs to an existing database of friends.
- Pipeline is 
	0. Camera Image
	1. Preprocess image (remove background)
	2. Detect face (sliding windows and learning algorithm).
		- Eyes segmentation
		- Nose segmentation
		- Mouth segmentation
	3. Logistic regression classifier (label for the identity of the person in the image)
- Ceiling analysis for this pipeline will be more involved.
- Assume the overall system has 85% accuracy (base)
- Manually remove the backgrounds of all images (through Photoshop) and then feed these images to the face detection block. This simulates a **perfect preprocessing** output. Observe improvement in accuracy. In this case 0.1%.
	- This shows that even if we had perfect background removal, it wouldn't make a huge difference in terms of overall system performance.
- Do the same for all other blocks. 
- Biggest improvement comes from preprocessing (removing background) to face detection (5.9%), so this should be the first priority.

### Thought Experiment
- Performing ceiling analysis on a pipelined ML system - plug in the ground truth labels for one component. 
- The performance of the system improves very little.
	- So probably not worth dedicating engineering resources to improving that system.
	- If component is a classifier, simply increasing the number of iterations for gradient descent (decreasing step size) will not cause it converge to better parameters.
	- Choosing more features for that component will not reduce bias.