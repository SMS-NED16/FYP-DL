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
- This is just the first step: we also need to form the bounding rectangle.  