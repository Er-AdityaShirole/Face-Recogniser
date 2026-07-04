# Used LBPH for recognition

This note is a complete, well-structured explanation of LBP, LBPH, training, recognition, scalability, and why modern face recognition systems use deep learning.

> This is written in a learning-friendly format so the topic flows naturally from basic concepts to advanced ideas.

---

## 1. Big Picture

Face recognition can be understood in a simple way:

Image → Features → Comparison → Label

In LBPH-based recognition:

- LBP is used to extract local texture features from an image.
- LBPH uses these features to represent a face.
- A new face is compared with stored face features.
- The closest match gives the predicted person.

So, LBPH is basically a handcrafted feature-based face recognition method.

---

## 2. What is LBP?

LBP stands for Local Binary Pattern.

It is not a face recognizer by itself. It is a texture descriptor.

### Main idea
LBP looks at a small neighborhood around each pixel and converts that local pattern into a number.

### Step-by-step process
1. Take a 3×3 neighborhood around a pixel.
2. Use the center pixel value as a threshold.
3. Compare each neighboring pixel with the center.
4. If a neighbor is greater than or equal to the center, assign 1.
5. Otherwise, assign 0.
6. Read the 8 bits in a fixed order.
7. Convert the binary pattern into a decimal number.

This decimal number is called an LBP code.

### Visual idea

Input pixel neighborhood:

```text
p1  p2  p3
p4  pc  p6
p7  p8  p9
```

Then compare each neighbor with `pc` and create a binary pattern.

```text
Neighbor >= center  -> 1
Neighbor < center   -> 0
```

So each pixel gets a code like:

```text
0 → 255
```

---

## 3. Important Clarification About LBP

This point is very important:

- The neighbor pixels are not permanently changed into 0s and 1s.
- The original grayscale image remains unchanged.
- Every pixel is later used as a center pixel one by one.
- The comparison always uses the original pixel intensity values.

### In simple words
LBP creates a new image where each pixel stores its LBP code, but it does not destroy the original image.

---

## 4. What Does an LBP Code Mean?

An LBP code does not represent the center pixel value.

It represents the texture around the center pixel.

### Meaning
- The center pixel only acts as a reference or threshold.
- The LBP code captures the local pattern around it.
- Similar textures usually produce similar LBP codes.

### Example idea
If two regions have similar shapes and brightness patterns, their LBP codes will be similar.

---

## 5. LBP Image

After computing LBP codes for every pixel, we get a new image called an LBP image.

### Properties of the LBP image
- It is a new image representation.
- It contains values from 0 to 255.
- It preserves the structure of the original image in a different form.

### Flow

```text
Original Image → LBP Image
```

---

## 6. Histogram Formation

Once we have an LBP image, we make a histogram.

### What a histogram stores
A histogram counts how many times each LBP code appears in the image.

### Example
If the LBP image contains these values:

```text
12 80 12
20 12 80
80 12 20
```

Then the histogram becomes:

```text
12 → 4 times
20 → 2 times
80 → 3 times
```

### Important point
A histogram has 256 possible bins because LBP codes range from 0 to 255.

---

## 7. Why Divide the Face into Grids?

A single histogram for the whole face is too general.

It does not preserve where features occur.

### Better idea
Divide the face into multiple smaller regions or grids.

Example:
- 8×8 grids
- 7×7 grids
- other grid sizes

### Why this helps
Each grid creates its own histogram.

This preserves spatial information.

That means the system knows not just what texture exists, but also where it exists in the face.

### Visual idea

```text
Whole face → one big histogram
Face divided into grids → many local histograms
```

### Benefit
This makes recognition more robust than using one single histogram for the entire face.

---

## 8. Feature Vector

The histograms from all grids are combined.

This combined vector is called the feature vector.

### Meaning
A feature vector is the final representation of one face image.

### Flow

```text
Grid 1 Histogram + Grid 2 Histogram + Grid 3 Histogram + ... = One Feature Vector
```

So, in LBPH:

```text
One face image → One feature vector
```

---

## 9. What Happens During LBPH Training?

Training in LBPH does not involve neural networks or machine learning in the modern sense.

### What happens
For every training image:

```text
Image → LBP Image → Grid Histograms → Feature Vector → Store (Feature Vector + Label)
```

### Important point
Training is basically:
- feature extraction
- storage
- labeling

### It does not involve
- weights
- neurons
- backpropagation
- gradient descent
- loss minimization

---

## 10. Does LBPH Learn Like a Neural Network?

No.

LBPH does not learn like deep learning models.

### It does not
- learn weights
- optimize parameters
- update a model during training
- use neural layers

### Instead, it simply
- extracts handcrafted features
- associates them with labels
- stores them

### Simple summary

```text
LBPH = Feature Extraction + Storage
```

---

## 11. One Person Has Multiple Feature Vectors

This is a very important concept.

If one person has many images, each image becomes one feature vector.

### Example
If Person A has 50 images:

```text
Image 1 → Feature Vector 1
Image 2 → Feature Vector 2
Image 3 → Feature Vector 3
...
Image 50 → Feature Vector 50
```

### Key idea
LBPH stores all of them separately.

It does not:
- average them
- merge them
- compress them into one single vector

---

## 12. Why Store Multiple Images Per Person?

Because different images capture different conditions.

### Examples of variation

| Condition | Why it matters |
|---|---|
| Lighting | Face appearance changes a lot under bright or dim light |
| Pose | Head angle changes the visible texture |
| Expression | Smiling, neutral, angry faces look different |
| Glasses | They change local appearance |
| Facial hair | Beard or mustache changes texture |
| Occlusion | Mask, sunglasses, or objects can hide parts of the face |
| Head rotation | Side view changes geometry and features |

### Why this is useful
More training images increase the chance that a future face will match one of the stored feature vectors closely.

---

## 13. How Recognition Works

When a new face comes in:

```text
New Face → LBP Image → Grid Histograms → Feature Vector
```

Then this new feature vector is compared with all stored feature vectors.

### Comparison step
It calculates distances between the new vector and the stored vectors.

### Example

| Comparison | Distance |
|---|---:|
| New vs A1 | 42 |
| New vs A2 | 18 |
| New vs A3 | 25 |
| New vs B1 | 130 |

The smallest distance is the best match.

So the system predicts:

```text
Recognized Person = Person A
```

---

## 14. Does LBPH Compare Every Stored Feature Vector?

Yes, in the straightforward implementation it does.

### Example
If you have:
- 100 people
- 50 images each
- total stored vectors = 5000

Then one recognition step may require comparison against all 5000 vectors.

### That means
The recognition time grows as the number of stored feature vectors grows.

---

## 15. Is LBPH Scalable?

LBPH works well for small or medium-sized systems.

### Good for
- attendance systems
- office recognition
- school systems
- small organizations

### Not ideal for
- large-scale databases
- millions of users
- very large real-time systems

### Why
Because comparing one input vector against a huge number of stored vectors becomes slow.

---

## 16. Scalability Example

| Scenario | Stored Feature Vectors | Comparison Cost |
|---|---:|---|
| 10 people × 50 images | 500 | Easy |
| 1000 people × 50 images | 50,000 | Slower |
| 50 million users × 20 images | 1 billion | Impractical |

### Conclusion
Direct comparison does not scale well for very large datasets.

---

## 17. Why Was LBPH Popular?

LBPH was popular because it was:
- simple
- lightweight
- easy to implement
- fast enough for small datasets
- useful for basic face recognition tasks

It was especially practical before deep learning became common.

---

## 18. Why Modern Systems Use Deep Learning

Modern face recognition systems use deep learning methods like:
- FaceNet
- ArcFace
- DeepFace

### Why
These systems learn rich face representations automatically.

Instead of handcrafted histograms, they generate embeddings.

### Example

```text
Face → Embedding (vector of numbers)
```

Embeddings of the same person are close together, while different people are farther apart.

---

## 19. What Are Face Embeddings?

A face embedding is a compact numerical representation of a face.

### Difference from LBPH
- LBPH uses handcrafted histograms.
- Deep learning uses learned features.

### Advantage
Embeddings are much more discriminative and robust.

---

## 20. Do Modern Systems Also Compare Vectors?

Yes.

They still compare vectors, but the vectors are much better.

### Difference

| Method | Representation | Comparison Type |
|---|---|---|
| LBPH | Handcrafted histograms | Simple distance comparison |
| Deep Learning | Learned embeddings | Similar vector comparison, but much stronger features |

---

## 21. How Do Large Companies Scale to Millions of Users?

They do not compare against every user one by one.

Instead, they use advanced search techniques such as:
- Approximate Nearest Neighbor (ANN)
- Vector databases
- Indexing structures like HNSW and IVF

### Idea
These methods avoid checking every stored vector exhaustively.

They quickly find the most likely matches.

### Simple analogy
Like using a dictionary:

```text
You do not read every page from A to Z to find a word.
You jump near the relevant section.
```

Vector search works similarly for embeddings.

---

## 22. Why Not Use the Same Optimization for LBPH?

You could use indexing methods for LBPH too.

But the bigger problem is that LBPH features themselves are weaker.

LBPH struggles more with:
- large pose changes
- different lighting
- occlusion
- aging
- facial hair changes

Deep learning models handle these variations much better.

That is why deep learning became the standard for large-scale recognition.

---

## 23. LBPH vs Deep Learning

| Aspect | LBPH | Deep Learning |
|---|---|---|
| Feature type | Handcrafted | Learned automatically |
| Training style | No parameter learning | Learns millions of parameters |
| Representation | Histogram-based | Embedding-based |
| Accuracy | Moderate | High |
| Robustness | Lower | Much higher |
| Speed on small systems | Good | More expensive to train |
| Scalability | Limited | Excellent for large-scale use |

---

## 24. Complete LBPH Pipeline

```text
Collect Training Images
        ↓
For each image:
        ↓
Compute LBP Image
        ↓
Divide image into grids
        ↓
Create histogram for each grid
        ↓
Concatenate histograms
        ↓
Create one feature vector
        ↓
Store (Feature Vector + Label)
```

### Recognition flow

```text
New Face
        ↓
Compute LBP Image
        ↓
Create grid histograms
        ↓
Create feature vector
        ↓
Compare with all stored feature vectors
        ↓
Find smallest distance
        ↓
Return predicted person label
```

---

## 25. Common Doubts and Clear Answers

### 1. Is LBP the same as LBPH?
No.
- LBP is the basic texture descriptor.
- LBPH is a face recognition approach that uses LBP-based histograms.

### 2. Does LBPH learn like a neural network?
No.
It does not learn weights or use backpropagation.

### 3. Does LBPH compare every stored vector?
Yes, in the basic approach it compares the new face with every stored feature vector.

### 4. Is it slow for large databases?
Yes.
The more stored vectors there are, the slower recognition becomes.

### 5. Why store many images of one person?
To capture different conditions such as lighting, pose, expression, and accessories.

### 6. Why is LBPH not used for huge systems now?
Because deep learning methods are more accurate and more scalable.

---

## 26. Final Summary

### LBPH in one sentence
LBPH is a simple face recognition method that converts a face into local texture-based histograms and compares them with stored feature vectors.

### Main takeaways
- LBP extracts local texture patterns.
- Histograms summarize those patterns.
- Grids preserve spatial information.
- Each face image becomes a feature vector.
- Recognition compares the new vector with stored vectors.
- LBPH is good for small systems, not huge databases.
- Modern systems use deep learning for better accuracy and scaling.

---

## 27. Short Revision Checklist

- What is LBP?
- What does an LBP code mean?
- How is a histogram formed?
- Why divide a face into grids?
- What is a feature vector?
- How does training work in LBPH?
- Why store multiple images per person?
- How does recognition work?
- Why does LBPH not scale well?
- Why do modern systems use deep learning?

---

If you want, I can also turn this into a more polished academic-style README with a stronger introduction, conclusion, and exam-style revision section.
