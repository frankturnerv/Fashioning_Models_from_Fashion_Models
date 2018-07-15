# Fashion_Models_from_Fashion_Models
Using Data Science to Learn About Fashion

## Problem Statement
The goal of this project is to gain insight into the world of fashion by studying the work of live fashion events through images. I present a method where I isolate clothing and fashion models in images using TensorFlow and OpenCV. I break down images into their component colors and create profiles for individual images, fashion shows, designers, and seasons. I attempt to predict the designer of an image using logistic regression classification and naive bays classifications. Finally, I’m able to use the statistics on images to identify designers that are near and far from one another using a cosine distance recommender algorithm 

## Why do I care?
Fashion is an multi trillion dollar industry globally that touches all of our lives, whether we want to admit it or not. Clothing is an outlet for personal expression, and even though we may have picked the clothes out ourselves, their design is the result and inspirations of huge amounts of man hours and jobs throughout the world. A few top designers lead the world in putting new fashion ideas forward, defining trends that percolate through the fashion industry. This project is an attempt to tease the most visible images put forth by top designers to try and get some insight into the ideas they put out into the world.

## Gathering Data
Vogue.com is the website for Vogue Magazine, a leading fashion magazine that has been running since 1892. Their website covers every major fashion show from at least the top designers and brands. I chose to use this website for my web scraping because of their extensive database. 

Vogue can have hundreds of designers featured per season. Additionally, the majority of the designers featuring their fashion lines on the website are not actually from fashion shows, but are “Lookbooks”, simply a collection of stylized photos to show off the clothing in the line. I decided to limit my approach to only Fashion Shows and only the top designers[1]. This required me to manually go through the Vogue Website and select which shows I would use. While a painful process, this provided a few advantages:
1. Lookbooks are often highly stylized, the images could include filters on the lens, or props that are not fashion.
2. Fashion show images are typically all framed the same, a model is front and center, one foot forward. All the images from a fashion show usually have identical backgrounds.
3. This labeling could be used later to algorithmically determine fashion shows from lookbooks.

Fashion lines are broken up by season and style. Typically, a designer will come out with multiple lines per year, usually at least a Spring and Fall Line. Additionally, a very popular season is the “Resort” line, which comes out in June during international fashion weeks. Additionally, non Resort shows are labeled: 
* Ready-to-Wear: (for clothes that could end up directly on shelves, 
* “Couture” : For high fashion, generally clothes that would never be worn
* Menswear : Typically Featuring only men, but often a show labeled menswear is identical to a ready to wear show that just features men. 

## Isolating Models with TensorFlow: 
While fashion show images are nice in that they are mostly images of the model, the majority of the pixels are background and even other people. While I intend to build my own TensorFlow Algorithm to optimize for fashion items, I did not want that to be the focus of the project. I instead used an existing TensorFlow Model called imageAI to detect the fashion model. [2]

imageAI still had the problem of detecting other humans that may be in the background (other fashion models or observers), but conveniently, the fashion models are the largest thing in the image. I kept the largest file produced by imageAI, assuming that to be the fashion model, and ignored the rest. Upon inspection of my dataset, algorithm has captured the majority of the fashion model every time.

## Removing Background with Canny Edge Detection:
After the Images were compressed, and the model framed via TensorFlow, about 40% of the area of the image is still the background. To solve this problem I used a method called Canny Edge Detection in OpenCV. By reducing Noise and Calculating Intensity Gradient, the algorithm labels the pixel containing the edge of an object in an image with a 1 and all other objects as a 0 based on the hysteresis thresholding defined by the user[3]. 

Canny edge detection creates a whole new image. Additionally processing, not provided by openCV, is required to actually get the image of the model in the foreground. Since the canny image shape exactly matches the original, I multiplied every pixel to the left of the first edge and every pixel to the right of the last edge by 0. This produced exactly the results I wanted, with very little of the background left in the image.

## Reducing Images to a few Pixels and Naming Colors
Every image still contained between 10,000 and 50,000 pixels. My goal is to interpolate actual fabric color from pixel color, and a even the smallest shadows can change a pixel color on a consistent fabric. To reduce the number of pixels extracted from each image, I used a machine learning technique called K-Means clustering [4] and extracted the top 20 colors from each image. This is a common technique used for image compression, but works very well for my needs here. 

Finally, I needed to have interpretable results. A pixel color is great, but if I would have a hard time telling someone the clothing is [255,0,0] instead of the clothing is red. I was actually very surprised to find color naming conventions on a computer to be a challenging task. XKCD.com [5] did a survey of over 200,000 people, showing them different colors and asked them to describe colors. The results were enlightening, hilarious, and surprisingly useful. Thankfully, he posted top 954 most common color names along with their hex coded RGB values[6], which I used for naming conventions in my project. To reduce the number of features, I actually chose to use the top 152 colors from this list, based entirely on the biggest box of Crayola crayons available containing 152 unique colors.

To map the colors in my clusters to the named colors, I thought of colors existing in a cube, who’s coordinates are their respective RGB values. I converted the hex values from XKCD to RGB, and used Pythagorean Theorem to compare the cluster center with every color in my dictionary from XKCD, and selected the color with the minimum distance to be the best of the name of the color cluster approximation. 


## Image Classification Techniques
The goal of modeling in this project is to see if it can be used to develop an intuition for fashion shows, seasons, or designers based on color. I chose to use logistic regression because it has the advantage of interpretability among variables and is a relatively straightforward modeling algorithm. The biggest downside to a Logistic Regression is that it tends to be very slow, so I added the Naive Bayes Classifier from scikit-learn as sort of a back up for comparison. I chose Naive Bayes because it is commonly used in modeling with natural language processing and sentiment analysis. I have done some NLP projects in the past and noticed that my “bag-of-colors” looked a lot like a “bag-of-words”, the palettes from fashion images  looked a lot documents, and my XKCD color dictionary could be the corpus. 

My models would attempt to predict, based on the color palette, which designer produced the image. I trained my models 3 times with three different representations of color.
1. Existence of Color - if a color exists in a palette, the color would receive a 1. All colors are weighted equally.
2. Frequency of Color - The color columns would add up to 1, the ratio of each color in the image.
3. TF-IDF - Divides the frequency from 2 by the inverse document frequency* Gives emphasis to colors that are unique to the image. 
Additionally, I made sure to randomize and stratify my train-test-split, so that images were represented from every season and designer.

Surprisingly, both models actually showed some predictive power. More surprising, both models performed best based on the existence of color, extra feature engineering made the models significantly worse. Naive Bayes could predict models based on existence of Color with 31.42% accuracy and Logistic Regression predicted designer with 38.99%. 

## Finding the most “different” images and designers
Even though TF-IDF vectorization yielded worse results, I actually like having it as a metric to use for the basis of a recommender system. The concept is simple, every image essentially gives colors a “rating”. I can calculate the distance using cosine similarity algorithm [7] for every image, fashion show, season, or designer, using color as the feature.

The results of implementing this algorithm were the most informing, and led me on a scavenger hunt through my images to find the shows my algorithm says were the most different. I also extracted the color palette from these shows and you can see them below

## Conclusion and Future Work

I presented a method gaining insight into fashion images. I built a model that isolates as best as possible a fashion model from an image and extracts color from the remainder of the image. I used Logistic Regression to predict designers based on information from the image, but more importantly that technique gave me insight to identifying colors of designers. I also built a recommender system using Cosine Similarity to determine which images are most “different”, which gave a map to look through the most interesting images.

There is still quite a lot of work to be done in this project. Here is what I will try next. If you have made it this far, I’m open to suggestions on how I can improve! Please reach out to me by E-Mail Frank@FrankTurnerV.com

### Webscraping
Currently, my web scraping algorithm saves images on my computer. The images themselves do nothing but take up space after I extract color from them, and they are easily accessible on the web. In my next iteration I will delete the image from my computer after it has been analyzed in the interest of saving space. 

### Isolating Clothing
Labeling a color was important for my sanity to make sure I was making visualizations correctly, but is actually not necessary to give it a human a name. Color is a human construct, and even a name can be the result of bias, marketing, or even gender.

### Compressing Images
Compressing Images is a good way to improve speed, but so much of fashion seems to be in the little details. In the future, I’d like to improve the speed of my algorithms without sacrificing fidelity of the images.

### More TensorFlow
Canny edge detection is useful for segmenting the model, but not great detecting objects. In the future, I would like to train a tensor flow model to detect objects like “dress”, “pants”, and “coat” and extract color from that. I expect I will even be able to remove skin tones in the same manner. 

### Better Directory System
I’m still new to this whole thing of working with computers and having thousands of items to keep track of and analyze. I really think my directory system could use some work to make looking up things easy and so that every image has one place with all the relevant information in it.


*TF-IDF = Term Frequency / Inverse Document Frequency where IDF = log_e(number of images/number of images where color_n appears)

### References

[1]http://us.fashionnetwork.com/news/The-20-best-fashion-shows-in-the-world-in-September-2017,876882.html#.W0pWcthKiMI

[2]https://github.com/OlafenwaMoses/ImageAI/tree/master/imageai/Detection

[3]http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_canny/py_canny.html

[4]http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_ml/py_kmeans/py_kmeans_opencv/py_kmeans_opencv.html

[5]https://blog.xkcd.com/2010/05/03/color-survey-results/

[6]https://xkcd.com/color/rgb/

[1]https://en.wikipedia.org/wiki/Cosine_similarity
