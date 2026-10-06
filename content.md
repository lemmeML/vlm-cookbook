## Before We Begin

Show a 5 year old a photograph of a dog and ask what color it is. She will look at it and say "brown" without thinking twice.

Now try to explain how she did it.

Somewhere between the light hitting her eyes and the word leaving her mouth, a grid of colored dots became _a dog_. Then a string of sounds became _a question about the dog_. Then the two met, and out came an answer. She has no idea how any of this happened. For most of the history of computing, neither did we.

For a long time computers could not really _see_. We had to spell out exactly what to look for: edges, corners, colors, textures. Neural networks changed that. Instead of writing the rules, we could show a model millions of examples and let it learn the patterns for itself. Computer vision went from recognizing simple shapes to recognizing objects, faces and, eventually, entire scenes.

Language followed a similar path, as models went from predicting the next word to understanding increasingly complex relationships between words, sentences and ideas. Things got interesting when we joined the two kinds of models together and let them work as one.

Today models can look at an image, read a question about it and answer in plain language, with remarkable detail and accuracy. We call them **Vision Language Models**, or VLMs.

So how does a model built around language suddenly get to see? Under the hood, a VLM is several components stitched together: something that sees, something that reasons in language, and a bridge that lets the two talk to each other.

A computer doesn't see a dog. It sees a few hundred thousand numbers arranged in a grid, and nothing in those numbers says "dog". That is where a **Vision Encoder** comes in. Its job is to cut the image into small square patches, the way you might cut a photograph into tiles. On their own the tiles mean little. A patch of brown fur could belong to a dog, a bear, or a carpet. Meaning only emerges when the patches can _talk to each other_, when the patch with the ear can ask the patch with the snout what it's looking at. The mechanism that makes this conversation possible is called **Attention**. It is the single most important idea in modern AI, and you will implement it yourself in the chapters ahead. Stack enough layers of attention together and you get an **Encoder**, a machine that turns pixels into understanding.

But seeing is only half the job. The model also has to "_read_", and it has to hold the image and the words in its head at the same time. Here we run into a puzzle: images and words are completely different kinds of things. How do you put a picture _inside_ a sentence? The answer comes in two parts. First, a **Processor** prepares the image and the prompt, leaving room in the text for the picture to sit. Then a small bridge translates what the vision encoder saw into the same language the words are written in. Once that is done, a picture is no longer a foreign object. It becomes just another part of the sentence.

Finally, the model has to "_speak_". For this we need a **Language Model**, a **Decoder** that reads the combined sequence of image and words and writes its answer one word at a time. Each word it chooses is informed by everything that came before, including what it saw. Then we load pretrained weights into the model we built, write the **inference** loop, show it a picture and ask it a question. And it answers!

#### Isn't that cool? A few lines of code, arranged in the right order, give a machine the ability to see, something a 5 year old does naturally.

---

You don't need to be an expert to make this journey. If you know that a neural network is built from layers, that a linear layer multiplies its input by a matrix and adds a bias, and that we train models by measuring a loss and nudging the weights through backpropagation, you have everything you need. Everything else is explained as we go.

Read a chapter, then close the book and open the code editor. Try to write the code _before_ you feel ready. When you get stuck, and you will, come back and reread. That moment of being stuck is not a sign that something is wrong. It happens when the idea is being carved into your mind.

Don't try to memorize anything here. Memorized code is forgotten by next week. Understood code is yours forever. If you understand _why_ each piece exists and what problem would appear if you took it away, you will find something strange happens: the code starts to write itself.

So let's start our journey with the most basic question: **how does a machine understand meaning?**

---

# Chapter 1. Turning Data into Numbers

A machine doesn't see or read the way we do. It only deals in numbers. So before it can understand anything, we have to turn that thing into numbers that can be compared and learned from. Once a picture and a sentence are both represented as numbers, we can ask a simple question: **how close should their representations be if they mean the same thing?**

> **Main Idea: A matching image and caption should give a big dot product and every mismatched pair should give a small one.**

Eventually we want a vision encoder whose output a language model can use. But before we can connect vision and language, the image encoder has to learn which visual concepts go with which words. A photo of a dog and the words "a brown dog" should mean the same thing, and since a neural network only understands numbers, that shared meaning has to show up in the numbers. So we need a way to turn a picture or a sentence into numbers we can compare.

### Embedding vectors

An embedding is a list of numbers that stands for something: a word, a sentence, an image or anything else. A list of numbers like this is called a **vector**, so embeddings are often called embedding vectors. If two things mean similar things, we want their vectors to point in similar directions.

The standard way to measure how similar two vectors are is the dot product. You multiply the numbers position by position and add them up:

$$\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{n} a_i b_i$$

where $a_i$ and $b_i$ are the $i$-th numbers of $\mathbf{a}$ and $\mathbf{b}$, and $n$ is how many numbers each vector holds.

The dot product also equals $|\mathbf{a}|,|\mathbf{b}|\cos\theta$, where $\theta$ is the angle between the vectors. It tells two tales: how long the vectors are and which way they point. To measure direction only, we first normalize each vector to length 1, so that $|\mathbf{a}| = |\mathbf{b}| = 1$ and the dot product becomes exactly $\cos\theta$ _(the cosine similarity)_.

If both vectors point the same way, the dot product is $1$. If they are perpendicular (orthogonal) it is $0$, and if they point in opposite directions it is $-1$.

_(CLIP, which we meet later in this chapter, always normalizes its vectors this way.)_

Now we need data that teaches the encoders to place an image vector and a caption vector close together when they describe the same thing, and far apart when they don't. That data has to tell us which images and which words belong together.

### A pile of pictures

![pile](fig/ch1-pile.svg)

Imagine you have a huge pile of pictures, and each picture comes with a short description. A photo of a dog with the caption "a brown dog on the grass".

You build two encoders. An image encoder turns a picture into a vector. A text encoder turns a caption into a vector. Each encoder ends with a linear projection into a shared space of the same dimension $d$, so both vectors have the same number of entries and you can take their dot product. Both are then normalized to length 1.

Now take a batch of $N$ pictures and their $N$ captions. Encode them all. You get $N$ image vectors and $N$ text vectors. Compute the dot product of every image with every caption. You get a table $S$ with $N$ rows and $N$ columns.

Row $i$ is image $i$, column $j$ is caption $j$, and the cell at row $i$ column $j$ is how similar the model thinks image $i$ and caption $j$ are.

Picture $i$ goes with caption $i$. So the correct pairs are the cells where the row number equals the column number, $S_{ii}$. That is the diagonal of the table. Every cell off the diagonal, $S_{ij}$ with $i \neq j$, is a wrong pair. There are $N$ right pairs and $(N^2 - N)$ wrong ones.

![contrastive-table](fig/ch1-contrastive-table.svg)

**Training both encoders so that the diagonal cells become large and every other cell becomes small is called contrastive learning. The model learns by contrasting the right pair against all the wrong ones.**

But the encoders start with random weights, so the diagonal starts no brighter than any other cell, and since nobody can set millions of weights by hand, the model has to **learn** them: measure how far the table is from what we want, nudge every weight a little toward better, and repeat for every batch.

Measuring how far off the table is takes a single number, big when the diagonal is dim and small when it shines. That number is the **loss**, and backpropagation tells every weight in both encoders which way to move to shrink it. So the question becomes: how do we turn an $N \times N$ table of similarities into one number?

### How CLIP turns this into a loss

[CLIP](https://openai.com/index/clip/), from OpenAI, is the model that made this kind of training famous. To see its loss you first need to know how language models are trained, because CLIP borrows the same trick.

A language model reads "leaning tower of" and has to guess the next word. It outputs one score for every word in its vocabulary. These raw scores are called logits, and we write them as $z_1, z_2, \dots, z_V$ where $V$ is the size of the vocabulary. We want the score for "pisa" to be the highest.

To compare scores with a correct answer, we first turn the scores into a probability distribution with a function called **softmax**. For each score it computes the exponential and then divides by the sum of the exponentials of all scores. The results are all positive and add up to 1, $\sum_k p_k = 1$.

$$p_k = \text{softmax}(z)_k = \frac{e^{z_k}}{\sum_{j=1}^{V} e^{z_j}}$$

where $p_k$ is the probability the model gives to word $k$, $z_k$ is that word's logit, and $j$ is a counter that runs over all $V$ words.

The **cross entropy loss** looks at the probability given to the correct answer and punishes the model when it is low. The correct answer is given as a label $y$, which is just the index of the right class. If "pisa" is word number 72 in the vocabulary, the label is $y = 72$ and the loss is

$$\mathcal{L} = -\log p_y$$

where $\mathcal{L}$ is the loss and $p_y$ is the probability the model gives to the correct word.

If the model gives the right word probability close to 1, $\log 1 = 0$ and the loss is almost zero. If it gives it a tiny probability, the log is a big negative number and the loss is large. This loss fits any problem with many candidates and exactly one right answer.

Now look back at our table $S$. Image 0 is compared against $N$ captions, and only caption 0 is right. That is the same kind of problem, so we can use the same loss.

CLIP does the same thing to its table. Take row 0. It holds the similarity of image 0 with every caption, $S_{0,0}, S_{0,1}, \dots, S_{0,N-1}$. Treat those $N$ numbers as scores over $N$ classes. The right class is caption 0, so the label is 0. For row 1 the label is 1. For row $i$ the label is $y_i = i$. So the labels for all rows are simply $0, 1, 2, \dots, N-1$, which is exactly what `np.arange(n)` produces.

CLIP does this once along the rows, which asks each image to pick its caption, and once along the columns, which asks each caption to pick its image. The final loss is the average of the two

$$\mathcal{L} = \frac{1}{2}\left(\mathcal{L}_{\text{img}} + \mathcal{L}_{\text{txt}}\right)$$

where $\mathcal{L}_{\text{img}}$ is the cross entropy averaged over the $N$ rows (every image picking its caption) and $\mathcal{L}_{\text{txt}}$ is the same averaged over the $N$ columns (every caption picking its image).

But there is a problem. Cosine similarities can never leave $[-1, 1]$, so the right caption can beat a wrong one by at most 2, and softmax cannot turn such small gaps into a confident answer. Take a perfect table, where the right caption scores $1$ and every wrong one scores $-1$. With CLIP's batch of $N = 32{,}768$, the right caption gets probability

$$p = \frac{e^{1}}{e^{1} + 32{,}767 \cdot e^{-1}} \approx 0.00023$$

and the loss is still about $8.4$. The encoders cannot possibly do better than this, yet the loss keeps punishing them.

To fix this, every cell of the table is multiplied by $e^{t}$ before the loss, where $t$ is a learned number called the **temperature**. The model now works with $S_{ij} \cdot e^{t}$, which lets it stretch the similarities far beyond $[-1, 1]$ and decide for itself how sharp the softmax should be.

![clip loss](fig/ch1-clip-loss.svg)

Here is the whole CLIP loss,`I_e` and `T_e` are the normalized image and text embeddings, each of shape $[N, d]$.

```python
logits = np.dot(I_e, T_e.T) * np.exp(t)     # the N x N table, stretched by the temperature
labels = np.arange(n)                       # row i should pick column i
loss_t = cross_entropy_loss(logits, labels, axis=0) # softmax across column, each caption picks its image 
loss_i = cross_entropy_loss(logits, labels, axis=1) # softmax across row, each image picks its caption
loss = (loss_i + loss_t) / 2
```

The loss is complete, and the temperature fixed our problem. But it quietly created a new one. CLIP starts with $e^{t} \approx 14.3$ and lets it grow up to $100$, so the logits entering softmax no longer sit between $-1$ and $1$. They can reach $100$, and softmax exponentiates them. Can a computer even hold $e^{100}$?

### Why softmax is dangerous and how to fix it

The exponential grows very fast. $e^{100}$ is roughly 26881171418161354484126255515800135873611118, a number with 44 digits. Computers store numbers in a fixed number of bits, usually 16 or 32, so every format has a largest value it can hold. A 16 bit float (FP16) cannot go past $65504$, and $e^{12}$ is already bigger than that. Once an exponential passes that limit it becomes infinity, infinity divided by infinity is NaN (not a number), and training breaks.

To fix it, notice that softmax is a fraction, and multiplying the top and the bottom of a fraction by the same number does not change it. So multiply both by $e^{-c}$ for some constant $c$. Because $e^{z_k} \cdot e^{-c} = e^{z_k - c}$, this is the same as subtracting $c$ from every score before the exponential

$$\frac{e^{z_k}}{\sum_j e^{z_j}} = \frac{e^{-c} e^{z_k}}{e^{-c}\sum_j e^{z_j}} = \frac{e^{z_k - c}}{\sum_j e^{z_j - c}}$$

The factor cancels and the output is identical.

![stable-softmax](fig/ch1-stable-softmax.svg)

Pick $c$ to be the largest score in the vector, $c = \max_j z_j$. Now the largest value becomes $e^{0} = 1$, and everything else is smaller than 1. Nothing can overflow. In code we write it as:

```python
import numpy as np

def naive_softmax(x):
    e = np.exp(x)
    return e / e.sum()
    
def stable_softmax(x):
    z = x - x.max()          # the largest score becomes 0
    e = np.exp(z)            # every value is now at most 1
    return e / e.sum()

x = np.array([1000.0, 999.0, 998.0])
with np.errstate(over="ignore", invalid="ignore"):
    print("naive  ", naive_softmax(x))
print("stable ", stable_softmax(x))
```

**Output:**

```text
naive  [nan nan nan]
stable [0.66524096 0.24472847 0.09003057]
```

So softmax is safe now. Is it perfect? Not quite. Look at what the trick needs. To compute the probability of a single cell, we first need $c = \max_j z_j$, the largest score in the _entire_ row, and then the sum $\sum_j e^{z_j - c}$, again over the entire row. Every probability depends on every other score in its row. For a language model predicting one word, that is fine. For CLIP's giant table, it becomes a bottleneck.

### The problem with CLIP at scale

Contrastive learning gets better as the batch grows (more on why below), so we want $N$ as large as the hardware allows. CLIP used $N = 32{,}768$, and the cost of its loss grows with $N$.

To compute softmax for one row, you must first find the maximum of the whole row, then exponentiate everything, then sum the whole row, $\sum_{j=1}^{N} e^{S_{ij}}$. A device computing that row needs the entire row in its memory. The same goes for columns and CLIP needs both.

This makes it hard to split the table across many GPUs, and it makes very large batch sizes painful. Large batches matter in contrastive learning because more wrong pairs give the model more to contrast against. With a batch of $N$ you get $(N^2 - N)$ wrong pairs, so doubling the batch roughly quadruples the negatives.

The root of the problem is softmax itself. Its denominator ties every cell in a row together (and, for the second loss, every cell in a column), so no cell can be scored on its own. That raises a question: do the cells really need to compete? What if we asked each pair on its own, _is this a match or not_?

### The sigmoid loss

A later model, [SigLIP](https://arxiv.org/abs/2303.15343), says no. Its **sigmoid loss** stops treating each row as a competition between $N$ captions and treats every single cell as its own yes or no question: **do this image and this caption match?**

This is a binary classification task, and the tool for it is the sigmoid function.

**Sigmoid takes any number and squashes it into a value between 0 and 1.**

![sigmoid](fig/ch1-sigmoid.svg)

$$\sigma(s) = \frac{1}{1 + e^{-s}}$$

where $\sigma$ (sigma) is the sigmoid and $s$ is any score, for us a single cell $S_{ij}$. A big positive $s$ gives something close to 1, a big negative $s$ gives something close to 0, and $s = 0$ gives exactly $0.5$.

Each cell gets a label $y_{ij}$ that is 1 on the diagonal and 0 everywhere else

$$y_{ij} = \begin{cases} 1, & \text{if } i = j \ 0, & \text{if } i \ne j \end{cases}$$

We push $\sigma(S_{ij})$ toward 1 for the diagonal and toward 0 for every other cell, using the usual binary cross entropy for each cell

$$\mathcal{L}_{ij} = -\Big[y_{ij}\log \sigma(S_{ij}) + (1 - y_{ij})\log\big(1 - \sigma(S_{ij})\big) \Big]$$

and the total loss is the average over all $N^2$ cells.

The formula looks busy, but only one of its two terms is ever switched on. On the diagonal $y_{ij} = 1$, the second term vanishes and the loss is $-\log \sigma(S_{ij})$, which is small only when $\sigma(S_{ij})$ is close to 1. Off the diagonal $y_{ij} = 0$, the first term vanishes and the loss is $-\log\big(1 - \sigma(S_{ij})\big)$, which is small only when $\sigma(S_{ij})$ is close to 0.

The real sigmoid loss also learns a bias $b$ next to the temperature, so the model can shift all scores at once. That helps at the start of training, when almost every cell is a negative.

A batch of 4 images and 4 captions gives a table of $4^2 = 16$ cells. The 4 diagonal cells are matches with label 1. The other $16 - 4 = 12$ are non matches with label 0. In general a batch of $N$ gives $N$ positives and $(N^2 - N)$ negatives.

```python
import torch
import torch.nn.functional as F

# sigmoid loss

def sigmoid_loss(img_emb, txt_emb, t, b):
    logits = img_emb @ txt_emb.T * t.exp() + b   # [N, N]
    labels = torch.eye(logits.size(0))           # 1 on the diagonal, 0 elsewhere

    return F.binary_cross_entropy_with_logits(logits, labels) # sigmoid + BCE in one stable call, averaged over all N^2 cells

# toy batch; every cell is yes/no
N, d = 4, 8
img = F.normalize(torch.randn(N, d), dim=-1)
txt_random = F.normalize(torch.randn(N, d), dim=-1)
txt_matching = img.clone()                     # a perfect encoder would give this
t, b = torch.tensor(2.3), torch.tensor(-3.0)

print("positives", N, "| negatives", N * N - N)
print("loss with random captions:", sigmoid_loss(img, txt_random, t, b).item())
print("loss with matching captions:", sigmoid_loss(img, txt_matching, t, b).item())
```

**Output:**

```
positives 4 | negatives 12
loss with random captions: 0.6071510910987854
loss with matching captions: 0.2847801446914673
```

The big win is independence. No cell needs to know about any other cell. Look at $\mathcal{L}_{ij}$ again, it only uses $S_{ij}$. There is no row maximum and no row sum. So you can cut the table into blocks, send each block to a different device and compute them separately. This is why the sigmoid loss scales to batches of a million pairs.

![sigmoid loss blocks](fig/ch1-sigmoid-loss-blocks.svg)

Now let's step back and look at what we have done so far. We set out to build a model that can see and talk, and so far we have trained two encoders against each other. What do we actually keep?

Our vision language model only keeps the image encoder from this training, not the text one. Why pick an encoder trained this way rather than a plain image classifier?

Because its image vectors were trained to line up with language. They already carry the kind of meaning text cares about. Also this training data is cheap. The internet is full of images with descriptions, like Wikipedia captions or the alt text of HTML images, which is the text shown when an image fails to load. Some descriptions are wrong or noisy, but with billions of examples the model still learns good representations.

![ch1-so-far](fig/ch1-so-far.svg)

That image encoder takes in a picture and produces a vector, and we still have no idea what happens inside. Time to build one ourselves.

---

# Chapter 2. Vision Transformer

At the end of the last chapter we left the image encoder as a black box: a picture goes in, a vector comes out. Now we open it. **The encoder we are going to build is a Transformer**, and a Transformer does not take images directly. So before we can build one for images, we must know what a Transformer expects to be fed.

### What a Transformer is, just enough

The Transformer was introduced in 2017, in the paper "[Attention Is All You Need](https://arxiv.org/abs/1706.03762)", to translate text from one language to another. You don't need its details yet, the upcoming chapters build it piece by piece. You only need its contract, the promise it makes about what goes in and what comes out.

A Transformer takes a list of vectors. Each item in the list is called a **token**. In text, a token is a word or a piece of a word. It becomes a vector through an **embedding table**, a big matrix with one row per token in the vocabulary. Every token has an id, a whole number, and the id simply picks its row. The word "dog" might be id 3290, so its vector is row 3290 of the table. The rows start random and are learned during training, just like the encoder weights in chapter 1.

The Transformer is a stack of identical **layers**. Each layer takes a list of $N$ vectors and returns a list of $N$ vectors, each of the same length as before. Nothing is added and nothing is removed. What changes is the content: every output vector is rewritten using information from the _other_ vectors in the list. The part of the layer that lets tokens look at each other is called **attention**. After enough layers, the vector for the word "bank" knows whether the sentence was about rivers or money.

We will write the shape of that list as $[B, N, D]$. A shape lists how many entries a tensor has along each axis, so $[B, N, D]$ is a 3D block of numbers: $B$ examples, each holding $N$ tokens, each token holding $D$ numbers. $B$ is the batch size (how many examples we process at once), $N$ is the number of tokens (code often calls it `seq_len`) and $D$ is the length of each token vector (the embedding dimension, called `hidden_size` in the config and `embed_dim` inside some modules; all three names mean the same number).

A Transformer where every token may look at every other token is called an **encoder**. That is what we build here. A Transformer where each token may only look at the tokens before it is called a **decoder**. That is the language model.

Notice what the contract does _not_ say: it never mentions words. A Transformer takes a list of vectors and returns a list of vectors. If we can turn an image into a list of vectors, the same machine should work.

So how do we turn an image into a list?

### Why not one token per pixel

An image arrives as a grid of pixel values, three numbers (red, green, blue, or RGB) for every pixel. The obvious move is to make every pixel a token.

But a $224 \times 224$ image has $50{,}176$ pixels, and attention compares every token with every other token. That is $50{,}176^2 \approx 2.5$ billion scores, for every layer. Stored as 32 bit floats, 4 bytes each, that one table alone takes about 10 GB, for a single image in a single layer. And a single pixel means almost nothing on its own anyway, it is just a dot of color.

So we need bigger pieces: a way to cut the image into a sequence short enough to afford, where each token is big enough to mean something.

### The Vision Transformer idea

Text is already a sequence. An image is a grid. The **Vision Transformer**, or **ViT**, from the paper "[An Image Is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)", turns the grid into a sequence by cutting the image into square patches and treating each patch as a token.

Our image encoder is a Vision Transformer.

The title of that paper is the whole idea. A $16 \times 16$ patch plays the role of a word. A $224 \times 224$ image cut this way gives $196$ patches instead of $50{,}176$ pixels, so attention computes $196^2 = 38{,}416$ scores instead of about 2.5 billion. Each patch holds $16 \times 16 = 256$ pixels, so there are 256 times fewer tokens. The number of scores grows with the _square_ of the number of tokens, so there are $256^2 = 65{,}536$ times fewer scores.

That is the paper's first message. The second is just as important: once the image is a sequence of patch vectors, **nothing else needs to change**. The layers are the same ones used for text.

### Say hello to ViT

Before we build the first piece, here is the whole machine, so you always know where the piece you are building fits.

![vision transformer](fig/ch2-vision-transformer.svg)

A $224 \times 224$ photo comes in. It is cut into a grid of $16 \times 16$ pixel patches, 14 across and 14 down, so 196 patches. One layer then looks at each patch and turns its pixels into a list of 768 numbers. Now we have 196 vectors, one per patch, and each one describes what its patch _looks like_. But cutting the image into a list lost track of where each patch was, so we add a second vector to each one that says which slot it came from.

The list then enters the big box in the middle, the **Transformer encoder**. It is a stack of identical layers, 12 in our config, and every layer does the same four things in the same order. A **norm** keeps the numbers at a steady scale, so they don't blow up or fade away as they pass through many layers (the next chapter). Then **attention**, where the patches finally talk: each patch looks at every other patch and borrows what is useful. Then another norm, and an **MLP**, a small network that works on each patch's vector by itself and digests what attention just gathered. (Each layer also has two shortcuts that add its input back to its output. We will meet them when we build the layer.) After the last layer comes one final norm, and out come 196 vectors.

Going in, the vector for a patch of brown fur only knew it was brown fur. Coming out, it has seen the ear, the snout and the grass, so it knows it is fur _on a dog_. Same patch, same slot, but its vector now carries its context.

You will also notice two greyed out boxes, **CLS** and **MLP head**. They belong to a version of the ViT that we don't build, and we come back to them right after this walk.

Here is the same walk again, with the shape of the data after every step.

```text
  image                      [B, 3, 224, 224]     batch, channels, height, width
  cut and embed patches      [B, 768, 14, 14]
  grid to sequence           [B, 196, 768]
  + position vectors         [B, 196, 768]
  12 encoder layers          [B, 196, 768]
  final norm                 [B, 196, 768]
```

The first shape puts the 3 color channels _before_ height and width. That is the order PyTorch layers expect. An image loaded from a file comes the other way round, as height, width, channels, so the processor we build later moves the channels to the front. Why the second shape puts 768 in front of the grid will be explained later in the chapter.

Look at the column of shapes. After the patches become a sequence, the shape never changes again. That is the Transformer contract at work, $N$ vectors in, $N$ vectors out. The 196 vectors that come out are the 196 that went in, each one rewritten by everything it learned from the others.

If you read about the Vision Transformer anywhere else, you will meet one difference: the number 197. The original ViT was built to classify images, "this is a dog". It borrowed a trick from a text model called BERT and put one extra learnable vector at the front of the sequence, called the **CLS** token, short for classification. It carries no patch. It travels through all the layers collecting information from the patches, and at the end only the CLS vector goes into a small classification head that outputs one score per class. That is $196 + 1 = 197$ tokens.

We don't need any of that. We are not classifying, we want the language model to see the whole image, patch by patch. So our encoder has no CLS token and no classification head. That is why they are greyed out in the figure. Our sequence has exactly 196 tokens.

### Configuring

Before we write any layer, we need one place to keep the numbers every layer will ask for: how long each token vector is, how big the image is, how big a patch is, how many layers to stack.

Every module you build reads its sizes from a config object. Start `vision_encoder.py` with it.

```python
class VisionConfig:
    def __init__(
        self,
        hidden_size=768,           # D, length of every token vector
        intermediate_size=3072,    # width inside the MLP
        num_hidden_layers=12,      # how many encoder layers are stacked
        num_attention_heads=12,    # attention heads per layer
        num_channels=3,            # red, green, blue
        image_size=224,            # images are resized to 224 x 224
        patch_size=16,             # each patch is 16 x 16 pixels
        layer_norm_eps=1e-6,       # small number that prevents division by zero
        attention_dropout=0.0,     # dropout inside attention, 0 means switched off
        num_image_tokens=None,     # image tokens the processor reserves, set from the real config
        **kwargs                   # swallows any extra keys from the pretrained config file
    ):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.num_image_tokens = num_image_tokens
```

These defaults are the sizes of the base model from the ViT paper. We use them throughout this chapter because they are easy to follow. The pretrained model we load later is several times bigger (vectors of length 1152, 27 layers, 16 heads), but you won't have to change a single line of code for it. Every module reads its sizes from this config, and when the real model is loaded, its own config file simply replaces these numbers.

The only difference worth remembering now is one number. The real model cuts its images into 14 pixel patches instead of 16. A 224 image then gives $224 / 14 = 16$ patches per side and $16 \times 16 = 256$ patches instead of 196. When you see 256 in a later chapter, that is where it comes from.

### Counting patches

Before we cut anything, how many tokens will we end up with?

If the image is $H = 224$ pixels wide and each patch is $P = 16$ pixels wide, you get $224 / 16 = 14$ patches across. The same number down. So $14 \times 14 = 196$ patches. (Our images are square, so $H$ stands for both the height and the width.)

The rule is

$$N_{\text{patches}} = \left(\frac{H}{P}\right)^2$$

where $H$ is the image size in pixels and $P$ is the patch size in pixels.

In code, `self.num_patches = (self.image_size // self.patch_size) ** 2`. The `//` is integer division, it throws away any remainder. If the image size were not a multiple of the patch size, the leftover strip of pixels at the edge would simply be ignored.

The token count grows fast with resolution. The model family we load also comes in a high resolution version that reads images of 896 pixels with the same 14 pixel patches. That is $896 / 14 = 64$ per side and $64^2 = 4096$ patches, so 4096 image tokens, 16 times more than at 224.

![patch embedding](fig/ch2-patch-embedding.svg)

### Look at your patches

Numbers are easy to nod along to. Before going further, cut a real photo into patches and look at them. Take any photo, resize it to $224 \times 224$ (a photo that is not square gets squashed a little, which is fine here), and slice it into its $14 \times 14$ grid with nothing but NumPy.

The only tricky line is the `reshape`. Any row index $h$ between 0 and 223 can be written as $h = 16a + b$, where $a$ (0 to 13) says which row of patches the pixel is in and $b$ (0 to 15) says which pixel row inside that patch. Reshaping the height axis of 224 into two axes of 14 and 16 splits every row index into exactly this $(a, b)$ pair. The same happens to the width. After that, the transpose brings the two "which patch" axes to the front, so `patches[r, c]` is one whole $16 \times 16$ tile.

```python
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

image = Image.open("dog.jpg").convert("RGB").resize((224, 224))
x = np.array(image)                                      # [224, 224, 3], height, width, channels
P = 16
n = 224 // P                                             # 14 patches per side

# split height into (14 blocks, 16 rows) and width into (14 blocks, 16 columns)
patches = x.reshape(n, P, n, P, 3)                       # [14, 16, 14, 16, 3]
patches = patches.transpose(0, 2, 1, 3, 4)               # [14, 14, 16, 16, 3], dims 1 and 2 swapped so each patch is whole

fig, axes = plt.subplots(n, n, figsize=(7, 7))
for r in range(n):
    for c in range(n):
        axes[r, c].imshow(patches[r, c])                 # patch number r * 14 + c
        axes[r, c].axis("off")
plt.subplots_adjust(wspace=0.05, hspace=0.05)
plt.show()
```

**Output:**

![dog-output](fig/dog-output.png)

You will see the photo with thin white gaps between 196 small tiles. Some tiles are obviously an eye or a nose. Most are just fur, grass or trees, and on their own could belong to almost anything. That is exactly what the encoder layers will have to fix later, by letting each tile ask the others what they see.

Now we know how many patches we want. How do we actually cut them out, and turn each one into a vector? The naive way would be to slice the image into 196 tiles like we just did, flatten each tile's $16 \times 16 \times 3 = 768$ numbers into a list, and multiply every list by the same weight matrix. It turns out one layer already does all of that in a single call.

### Cutting patches with a convolution

A 2D convolution slides a small window, called the kernel, across an image. At each position it multiplies the pixels under the window by learned weights, adds them up and adds one learned bias, giving one number. Color images have three channels, red, green and blue, so the kernel covers all three at once. A $16 \times 16$ kernel over 3 channels looks at $16 \times 16 \times 3 = 768$ pixel values and turns them into one number. A layer can hold many kernels, each producing its own output channel, so with 768 output channels you get 768 numbers for every window position.

The two 768s here are a coincidence of this config. The first is how many pixel values one window sees, the second is `hidden_size`, the length we want each token vector to have. With 14 pixel patches a window sees $14 \times 14 \times 3 = 588$ values, but the output length is still whatever `hidden_size` says.

The stride is how far the window jumps between positions. If the kernel is 16 and the stride is also 16, the window jumps exactly one patch at a time. No overlap and no gaps. Each window position sees exactly one patch and turns it into 768 numbers. That is exactly what we wanted, one vector per patch, and it is called a **patch embedding**.

![kernel and stride](fig/ch2-kernel-stride.svg)

The padding is set to `"valid"`, which means no extra border is added around the image. Since 224 divides evenly by 16, there is nothing left over at the edges that would need a border.

So `nn.Conv2d(in_channels=3, out_channels=768, kernel_size=16, stride=16, padding="valid")` takes an input of shape $[B, 3, 224, 224]$ and gives $[B, 768, 14, 14]$. The output is still a grid, 14 by 14, where each cell holds a 768 long vector.

### Convolution by hand

That description is a lot of words for a simple operation. Let's do one small enough to compute in your head. A grayscale image, so 1 channel, of $4 \times 4$ pixels, a $2 \times 2$ kernel, stride 2 and no bias.

$$\text{image} = \begin{pmatrix} 3 & 5 & 1 & 9 \\ 2 & 7 & 4 & 6 \\ 8 & 0 & 5 & 3 \\ 1 & 6 & 2 & 4 \end{pmatrix} \qquad \text{kernel} = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$$

The window starts at the top left and sees the patch $\begin{pmatrix} 3 & 5 \\ 2 & 7 \end{pmatrix}$. Multiply each pixel by the kernel weight in the same spot and add everything up

$$3 \cdot 1 + 5 \cdot 0 + 2 \cdot 0 + 7 \cdot (-1) = 3 - 7 = -4$$

Jump 2 pixels right. The window sees $\begin{pmatrix} 1 & 9 \\ 4 & 6 \end{pmatrix}$ and gives $1 - 6 = -5$. Jump down to the bottom left, $\begin{pmatrix} 8 & 0 \\ 1 & 6 \end{pmatrix}$ gives $8 - 6 = 2$. Bottom right, $\begin{pmatrix} 5 & 3 \\ 2 & 4 \end{pmatrix}$ gives $5 - 4 = 1$.

Four patches, four numbers, and they land in a $2 \times 2$ grid in the same arrangement as the patches they came from

$$\text{output} = \begin{pmatrix} -4 & -5 \\ 2 & 1 \end{pmatrix}$$

![convolution by hand](fig/ch2-conv-by-hand.svg)

(If you met convolution in a signal processing class, you may remember the kernel being flipped first. Deep learning libraries skip the flip, strictly that is called cross correlation, but everyone calls it convolution. Since the weights are learned, the flip would make no difference.)

PyTorch gives the same result.

```python
import torch
import torch.nn as nn

img = torch.tensor([[3., 5., 1., 9.],
                    [2., 7., 4., 6.],
                    [8., 0., 5., 3.],
                    [1., 6., 2., 4.]]).reshape(1, 1, 4, 4)       # [B, C, H, W]

conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, stride=2, bias=False)
conv.weight.data = torch.tensor([[1., 0.],
                                 [0., -1.]]).reshape(1, 1, 2, 2)  # [out_channels, in_channels, K, K]
print(conv(img).detach())
```

**Output:**

```text
tensor([[[[ -4., -5.],
          [  2.,  1.]]]])
```

This toy has one output channel, so each patch became one number. Our real layer has 768 output channels, 768 different kernels, so each patch becomes 768 numbers, one from each kernel. That is the patch vector.

### A convolution is a linear layer in disguise

Look at the first patch again. Read its pixels row by row into a list, $[3, 5, 2, 7]$, and do the same with the kernel, $[1, 0, 0, -1]$. The dot product of the two lists is

$$3 \cdot 1 + 5 \cdot 0 + 2 \cdot 0 + 7 \cdot (-1) = -4$$

the same number the convolution produced. That is not a coincidence. When the stride equals the kernel, every window sees its own patch and nothing else, so the convolution is exactly this: flatten each patch into a list, and multiply it by a weight matrix, the same matrix for every patch, then add the bias. With 768 kernels, the weight matrix has one row per kernel. In other words, a linear layer applied to every flattened patch.

This matters because the ViT paper describes the patch embedding this way. It says it will "flatten the patches and map to D dimensions with a trainable linear projection", with no convolution in sight. Our code uses a convolution. They are the same computation written two ways, and here is the proof. Cut the patches by hand, multiply them by the convolution's own weights reshaped into a matrix, add its bias, and compare.

```python
conv = nn.Conv2d(in_channels=3, out_channels=768, kernel_size=16, stride=16)
x = torch.randn(1, 3, 224, 224)

# the convolution way
out_conv = conv(x).flatten(2).transpose(1, 2)                     # [1, 196, 768]

# the paper way, flatten every patch and multiply by one matrix
patches = x.unfold(2, 16, 16).unfold(3, 16, 16)                  # [1, 3, 14, 14, 16, 16], windows along height (dim 2) then width (dim 3)
patches = patches.permute(0, 2, 3, 1, 4, 5)                      # [1, 14, 14, 3, 16, 16], each patch's channels and pixels together
patches = patches.reshape(1, 196, 3 * 16 * 16)                   # [1, 196, 768], one flat list per patch
W = conv.weight.reshape(768, -1)                                 # [768, 768], one row per kernel
out_linear = patches @ W.T + conv.bias                           # [1, 196, 768]

print(torch.allclose(out_conv, out_linear, atol=1e-4))
```

**Output:**

```text
True
```

(`flatten(2).transpose(1, 2)` on the first line turns the conv's grid into a list of 196 vectors so the two results can be compared. We explain it properly later in the chapter.)

![convolution is linear](fig/ch2-conv-is-linear.svg)
The code below splits our dog image into 196 patches from earlier and shows them as sequence and the matrix form.

```python
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

image = Image.open("dog.jpg").convert("RGB").resize((224, 224))
x = np.array(image)                                      # [224, 224, 3]
P = 16
n = 224 // P                                             # 14

patches = x.reshape(n, P, n, P, 3).transpose(0, 2, 1, 3, 4)   # [14, 14, 16, 16, 3]
seq     = patches.reshape(n * n, P, P, 3)                     # [196, 16, 16, 3], k = r * 14 + c
flat    = seq.reshape(n * n, -1)                              # [196, 768], one row per patch 

fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(x)

for i in range(1, n):
    ax.axhline(i * P, color="white", lw=0.5)
    ax.axvline(i * P, color="white", lw=0.5)
for r in range(n):
    for c in range(n):
        ax.text(c*P + P/2, r*P + P/2, r*n + c, color="yellow", fontsize=6, ha="center", va="center")
ax.set_title("image → 14×14 = 196 patches"); ax.axis("off")
plt.show()

# first 28 patches as a sequence (rows 0 and 1 of the grid)
fig, axes = plt.subplots(1, 28, figsize=(20, 1.5))
for k in range(28):
    axes[k].imshow(seq[k]); axes[k].set_title(k, fontsize=7); axes[k].axis("off")
plt.suptitle("sequence order: patch 0, 1, 2, ... (row by row)", y=1.15)
plt.show()

# the whole sequence as a matrix: 196 patches × 768 pixel values
plt.figure(figsize=(10, 4))
plt.imshow(flat, aspect="auto", cmap="gray"
plt.xlabel("768 values = 16 × 16 × 3"); plt.ylabel("patch index 0–195")
plt.title("[196, 768]: input to the patch embedding")
plt.show()

print(x.shape, "→", patches.shape, "→", seq.shape, "→", flat.shape)
```

**Output:**

![dog-table](fig/dog-table.png)
![dog-seq](fig/dog-sequence.png)
![dog-emb](fig/dog-emb.png)

```text
(224, 224, 3) → (14, 14, 16, 16, 3) → (196, 16, 16, 3) → (196, 768)
```

So why use the convolution at all? Because it is one line, it cuts and multiplies in a single call, and it is heavily optimized on GPUs. And because the pretrained weights we load are stored in this shape.
### Why not a CNN?

We just used a convolution to cut patches, and before 2020 most of the best vision models were **convolutional neural networks**, or CNNs, built from many stacked convolutions. So why not use a whole CNN as the image encoder?

A CNN stacks many convolutions with small kernels, typically $3 \times 3$, that slide one pixel at a time, so the windows overlap heavily. Each layer only looks at a small neighborhood of the layer below. For the dog's ear to relate to its tail, information has to travel there layer by layer, the view widening a little each time.

That design bakes in two assumptions. Nearby pixels belong together, and a pattern means the same thing wherever it appears in the image. Assumptions built into a model's design, instead of learned from data, are called **inductive bias**. For images they are usually true, so a CNN learns well from relatively little data.

A ViT throws almost all of that away. Its only image specific parts are the patch cutting and the position vectors. From the very first layer, attention lets any patch look at any other patch, near or far. The price is that the model has to _learn_ from data that neighbors usually matter, because nobody told it.

![cnn vs vit](fig/ch2-cnn-vs-vit.svg)

The ViT paper measured this trade. Trained on ImageNet alone, about 1.3 million images, ViT lost to strong CNNs. Pretrained on JFT-300M, a private Google dataset of about 300 million images, it beat them, and the largest ViTs gained the most from the extra data.

Now remember chapter 1. Our encoder learns from billions of image and caption pairs taken from the web. Data is the one thing we have in abundance, and that is where ViT wins.

One thing not to confuse. Our patch layer is a convolution, but it is not a CNN. It is a single layer whose windows never overlap. It only cuts the image and projects each patch. All the seeing is left to the Transformer.

### When windows overlap

What would happen if ours did overlap, the way a CNN's windows do? That happens whenever the stride is smaller than the kernel. For image size $H$, kernel $K$ and stride $S$, the number of positions that fit along one side is

$$\left\lfloor \frac{H - K}{S} \right\rfloor + 1$$

where the $\lfloor \cdot \rfloor$ brackets mean "round down to a whole number", since a window that only half fits is not counted. The $+1$ counts the very first window, the one at the left edge before any jump.

With $K = 16$ and $S = 8$ on a 224 image that is $\frac{224 - 16}{8} + 1 = 26 + 1 = 27$. So $27 \times 27 = 729$ patches, each sharing half its pixels with its neighbor. More tokens, more compute. When $S = K$ the formula becomes $\frac{H - K}{K} + 1 = \frac{H}{K}$, which is our $H / P$ again.

![kernel and stride](fig/ch2-kernel-stride.svg)

(The figure compares stride 8 and stride 4 along one row of pixels. In 2D both sides count, so a 224 image with kernel 16 and stride 8 gives $27 \times 27 = 729$ patches.)

Overlap is what gives a CNN its slowly widening view, but in our patch layer it would only multiply the tokens, and attention pays for every token squared: 729 tokens instead of 196 means about 14 times more scores. So our patch layer keeps the stride equal to the kernel.

### From grid to sequence

Back to our patch layer. The convolution gave us one vector per patch. But look at the shape, $[B, 768, 14, 14]$. The patches are still arranged as a $14 \times 14$ grid, and the 768 numbers of each patch come _before_ the grid, because a convolution always puts its output channels right after the batch. That is not the list of tokens a Transformer expects.

The Transformer wants a list, not a grid. Number the dimensions of $[B, 768, 14, 14]$ as 0, 1, 2, 3. `flatten(2)` merges every dimension from index 2 onward into one, so $14 \times 14$ becomes $196$. The shape $[B, 768, 14, 14]$ becomes $[B, 768, 196]$. The grid is read row by row, so patch $(r, c)$ lands at position $14r + c$.

But now the embedding dimension (dimension 1) comes before the patch dimension (dimension 2). A Transformer expects one row per token, shape $[B, \text{tokens}, \text{embedding}]$. So we swap dimensions 1 and 2 with `transpose(1, 2)` and get $[B, 196, 768]$.

![flatten and transpose](fig/ch2-flatten-transpose.svg)

Now we have a sequence.

### Adding position

Flattening throws away where each patch was. Patch 0 was top left, patch 195 was bottom right, but the list does not say so.

Does that matter? Yes. Attention, which we build later, treats its input like a bag of tokens. Shuffle the patches and every patch gets exactly the same result, just in the shuffled order. Without positions, a picture and the same picture with its tiles scrambled would look identical to the model.

We fix this by adding a position vector to each patch

$$\mathbf{e}_i = \mathbf{x}_i + \mathbf{p}_i$$

where $\mathbf{x}_i$ is the embedding of patch $i$, $\mathbf{p}_i$ is the position vector for slot $i$, and $\mathbf{e}_i$ is what finally enters the Transformer. Since we add them, $\mathbf{p}_i$ must have the same length as $\mathbf{x}_i$, 768 numbers. A position is not a single number here, it is a whole vector.

These position vectors are not computed by a formula. They are learned. We create `nn.Embedding(num_positions, embed_dim)`, the same kind of embedding table we met at the start of this chapter, except the ids are now slot numbers instead of word ids. It holds one learnable 768 long vector per slot, a matrix of shape $196 \times 768$. Give it slot number $i$ and it hands back row $i$. Position vector 0 is always added to patch 0, vector 1 to patch 1, and so on. During training the model shapes these vectors into whatever helps it know where a patch sits.

To look them all up at once we need the numbers 0 to 195. We store them with `self.register_buffer("position_ids", torch.arange(num_positions).expand((1, -1)), persistent=False)`. A buffer is a tensor that belongs to the module and moves to the GPU with it, but is not a trainable weight. `expand((1, -1))` gives it a batch dimension of size 1, shape $[1, 196]$, where $-1$ means "keep this dimension as it is". `persistent=False` means it is not saved with the weights, since it is just the numbers 0 to 195 and can be rebuilt any time. Then the forward adds `self.position_embedding(self.position_ids)`, shape $[1, 196, 768]$, to the patches. The batch dimension of 1 is copied across all $B$ images automatically, which PyTorch calls broadcasting, so every image gets the same position vectors.

```python
    self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
    self.register_buffer(
        "position_ids",
        torch.arange(self.num_positions).expand((1, -1)),   # [1, 196], the numbers 0 to 195
        persistent=False,
        )

    def forward(self, pixel_values):
        patch_embeds = self.patch_embedding(pixel_values)   # [B, D, 14, 14]
        embeddings = patch_embeds.flatten(2)                # [B, D, 196], dims 2 and 3 (the grid) merged into one
        embeddings = embeddings.transpose(1, 2)             # [B, 196, D], dim 1 (D) swapped with dim 2 (patches)
        embeddings = embeddings + self.position_embedding(self.position_ids)   # [B, 196, D], the [1, 196, D] positions broadcast over the batch
        return embeddings
```

### What the position vectors learn

Something strange should bother you here. The model is only ever given slot numbers, 0 to 195, in a line. Nobody tells it that slot 14 sits directly below slot 0, or that slot 13 is at the right edge and slot 14 back at the left. As far as the model knows, it is reading a sentence.

Yet the ViT paper found that after training, the position vectors of patches in the same row became similar to each other, and so did those in the same column, and nearby patches ended up more similar than distant ones. The model rediscovered the 2D grid on its own, from nothing but a list of slots, because knowing which patches are neighbors helps it see. The authors also tried handing the model 2D coordinates directly, and it did not do meaningfully better.

The trained weights of the paper's base model shows its sizes are exactly as our `VisionConfig` defaults: vectors of 768, patches of 16, images of 224, so a $14 \times 14$ grid of slots. Its position table is the trained version of the table we just described. Hugging Face stores it as a plain tensor instead of an `nn.Embedding`, but the numbers mean the same thing.

One difference. This model was built to classify, so it has the CLS token we talked about earlier, and its table holds $196 + 1 = 197$ rows. Row 0 belongs CLS, which doesn't correspond to any position in the image, so we drop it and keep the 196 patch slots.

For every slot, the code below computes the cosine similarity between that slot's position vector and every other slot's, and draws the result as a small heat map, placed where that slot sits in the image. You need `pip install transformers`, and the first run downloads about 350 MB of weights.

```python
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from transformers import ViTModel

vit = ViTModel.from_pretrained("google/vit-base-patch16-224", add_pooling_layer=False)

with torch.no_grad():
    pos = vit.embeddings.position_embeddings[0, 1:]     # [197, 768] -> [196, 768], row 0 is CLS, dropped
pos = F.normalize(pos, dim=-1)                          # length 1, so the dot product is the cosine
sim = pos @ pos.T                                       # [196, 196], every slot against every slot
side = int(sim.shape[0] ** 0.5)                         # 14 slots per side
sim = sim.reshape(side, side, side, side)               # [row, col] of one slot, then [row, col] of every other slot

fig, axes = plt.subplots(side, side, figsize=(8, 8))
for r in range(side):
    for c in range(side):
        axes[r, c].imshow(sim[r, c], vmin=-0.2, vmax=0.8, cmap="Oranges")   # how similar slot (r, c) is to every slot
        axes[r, c].axis("off")
plt.show()
```

If you see a message about unused `classifier` weights, that is the classification head we don't need, the same one greyed out in the figure at the start of this chapter.

![position similarity](fig/ch2-position-similarity.svg)

Each small map is one slot, and the dark dot marks where that slot sits. Wherever the map lights up, that slot's position vector is similar to the vector of the slot being lit, which is the model's way of saying the two are close. Look at any map and you will see a cross: its own row and its own column light up. On average, a slot's similarity to slots in its own row or column is 0.46, and to every other slot it is −0.06. Nothing told the model that 196 slots form a $14 \times 14$ grid. It learned that from the images.

### The full module

So the full forward is conv, flatten, transpose, add positions.

Here is the whole module.

```python
import torch
import torch.nn as nn

class VisionEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,          # stride = kernel, patches never overlap
            padding="valid",
        )
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),   # [1, 196], the numbers 0 to 195
            persistent=False,
        )

    def forward(self, pixel_values):
        patch_embeds = self.patch_embedding(pixel_values)   # [B, D, 14, 14]
        embeddings = patch_embeds.flatten(2)                # [B, D, 196], dims 2 and 3 (the grid) merged into one
        embeddings = embeddings.transpose(1, 2)             # [B, 196, D], dim 1 (D) swapped with dim 2 (patches)
        embeddings = embeddings + self.position_embedding(self.position_ids)   # [B, 196, D], the [1, 196, D] positions broadcast over the batch
        return embeddings
```

The shapes in the comments use our config. With the real weights every 14 becomes 16 and every 196 becomes 256, and the code does not change.

```python
config = VisionConfig()
embeddings = VisionEmbeddings(config)
x = torch.randn(1, 3, 224, 224)       # one random "image"
print(embeddings(x).shape)
```

**Output:**

```text
torch.Size([1, 196, 768])
```

We can now pass the patches through the embedding layer, using pretrained ViT weights loaded into our `VisionEmbedding` class. The code shows three steps: each patch becomes a 768-dimensional content vector, a learned position vector is added, and the sum is what the transformer receives.

```python
import torch
import torch.nn.functional as F
import torch.nn as nn
from types import SimpleNamespace
from transformers import ViTModel
import matplotlib.pyplot as plt

# Definition of VisionConfig class
class VisionConfig:
    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        image_size=224,
        patch_size=16,
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        num_image_tokens=None,
        **kwargs
    ):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.num_image_tokens = num_image_tokens

# Definition of VisionEmbeddings class
class VisionEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False,
        )

    def forward(self, pixel_values):
        patch_embeds = self.patch_embedding(pixel_values)
        embeddings = patch_embeds.flatten(2)
        embeddings = embeddings.transpose(1, 2)
        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings

vit = ViTModel.from_pretrained("google/vit-base-patch16-224", add_pooling_layer=False)

cfg = SimpleNamespace(hidden_size=768, image_size=224, patch_size=16, num_channels=3)
emb = VisionEmbeddings(cfg).eval()
with torch.no_grad():                                   # copy ViT weights into our module
    emb.patch_embedding.weight.copy_(vit.embeddings.patch_embeddings.projection.weight)   # [768, 3, 16, 16]
    emb.patch_embedding.bias.copy_(vit.embeddings.patch_embeddings.projection.bias)       # [768]
    emb.position_embedding.weight.copy_(vit.embeddings.position_embeddings[0, 1:])        # [196, 768], CLS dropped

pix = torch.tensor(x).permute(2, 0, 1).float().div(255).sub(0.5).div(0.5)[None]   # [1, 3, 224, 224], ViT normalisation

with torch.no_grad():
    tokens = emb.patch_embedding(pix).flatten(2).transpose(1, 2)[0]   # [196, 768], content only
    pos    = emb.position_embedding.weight                            # [196, 768], location only
    out    = emb(pix)[0]                                              # [196, 768], content + location

print("out == tokens + pos:", torch.allclose(out, tokens + pos, atol=1e-5))
print(f"avg norm  patch: {tokens.norm(dim=-1).mean():.1f}   position: {pos.norm(dim=-1).mean():.1f}")

# the three [196, 768] matrices
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for a, m, t in zip(ax, [tokens, pos, out], ["patch embedding", "+ position embedding", "= final embedding"]):
    a.imshow(m.detach().cpu().numpy(), aspect="auto", cmap="coolwarm", vmin=-3, vmax=3)
    a.set_title(f"{t}  [196, 768]"); a.set_xlabel("768 dims"); a.set_ylabel("patch index")
plt.tight_layout(); plt.show()

# pick one patch and ask: which slots look like it?
r, c = 7, 7                                             # move this onto the dog
k = r * n + c
def sim_map(v):
    v = F.normalize(v, dim=-1)
    return (v @ v[k]).view(n, n)                        # cosine of patch k with all 196, back to 14×14

fig, ax = plt.subplots(1, 4, figsize=(16, 4))
ax[0].imshow(x); ax[0].add_patch(plt.Rectangle((c*P, r*P), P, P, fill=False, ec="red", lw=2))
ax[0].set_title(f"patch {k}")
for a, m, t in zip(ax[1:], [tokens, pos, out], ["patch only: similar content", "position only: nearby slots", "patch + position"]):
    a.imshow(sim_map(m).detach().cpu().numpy(), cmap="Oranges"); a.set_title(t)
for a in ax: a.axis("off")
plt.show()
```

**Output:**
![patch+pos](fig/patch+pos.png)
![similar](fig/similar.png)

The patch embedding alone knows what is in a patch but not where, so similar-looking patches match anywhere in the image. The position embedding alone knows where but not what, so it matches nearby slots regardless of content. Adding them gives each of the 196 tokens both, and that $[196, 768]$ matrix is the input to the transformer layers.

### Where these vectors are going

In the first chapter we said a picture goes into the image encoder and a vector comes out. That was true when the Image Encoder was being trained for image-text alignment. To compare an image with a caption you need one vector per image, so during that training all the patch vectors are squeezed into a single vector by a small pooling layer at the end.

Our vision language model skips that squeeze. It keeps every patch vector that comes out of the encoder, resizes them to the language model's width with a single linear layer, and places them into the prompt as image tokens. To the language model, each patch really does become a word.

We now have 196 vectors. Each one knows what its patch looks like and where it sits. But each patch still knows nothing about any _other_ patch. The ear does not know there is a snout next to it. The encoder layers ahead will fix that by mixing the patches, and we are about to stack twelve of them.

![ch2-so-far](fig/ch2-so-far.svg)

The moment we stack that many layers, a new problem appears. The numbers flowing through them can drift in scale from layer to layer and from batch to batch, and training starts to wobble. So before we build these layers, we need a way to keep the numbers steady.

---
# Chapter 3. Keeping the Numbers Steady

At the end of the last chapter we had 196 patch vectors, each 768 numbers long, ready to pass through a stack of twelve encoder layers. But there is a problem, as those numbers pass through layer after layer, their scale can drift. Some layers may make them larger, others smaller, and training starts to wobble. So before we build the encoder layer, we need to build the small piece that keeps this drift under control: the **norm layer** you saw in the ViT figure in the last chapter.

> **Main Idea: Before every layer, shift and rescale each token's vector so its numbers have a mean of 0 and a spread of 1.**

Think of it as a volume knob. Each layer passes its numbers on to the next one. If one layer whispers and the next one shouts, nobody can follow the conversation. The norm brings every voice back to the same level before it reaches the next layer.

### Why the numbers drift

For a deeper dive into [Normalization] check our previous lessons. 

Let's start with the smallest piece of a layer: a single neuron in a linear layer. A neuron is a tiny calculator. It multiplies each input number $x_i$ by its corresponding weight $w_i$, adds all those products together (that is the dot product of the input $\mathbf{x}$ with the weight vector $\mathbf{w}$), and then adds bias '$b$'.

$$y = \mathbf{w} \cdot \mathbf{x} + b$$

here, $y$ is the output of the neuron.

Let's try it. Take $\mathbf{w} = [0.5, -1, 2]$, $b = 1$ and the input $\mathbf{x} = [1, 2, 3]$. Then

$$\mathbf{w} \cdot \mathbf{x} = 0.5 - 2 + 6 = 4.5$$

$$y = 4.5 + 1 = 5.5$$

Now double the input to $[2, 4, 6]$. The dot product doubles as well:

$$\mathbf{w} \cdot \mathbf{x} = 1 - 4 + 12 = 9$$

$$y = 9 + 1 = 10$$

The dot product follows the scale of the input, the bias doesn't chage, so the full output simply does not double. The same idea shows up in the gradient.

Training works by nudging the weights. But which way should each weight move? Backpropagation answers that by asking how much the output $y$ changes when weight $w_i$ changes by a tiny amount. The answer is the input value that weight multiplies:

$$\frac{\partial y}{\partial w_i} = x_i$$

Each weight's gradient contains its corresponding input $x_i$ as a factor. Double the input and you double the gradient. The step that gradient-descent step doubles with it:

$$\mathbf{w} \leftarrow \mathbf{w} - \eta , \frac{\partial \mathcal{L}}{\partial \mathbf{w}}$$

where $\eta$ (eta) is the learning rate, which controls the size of each step, and $\mathcal{L}$ is the loss.

![neuron-input](fig/ch3-neuron-input.svg)

Now put this neuron somewhere in the middle of a network that is being trained. Its input comes from the previous layer. After every batch, that layer updates its weights, so the numbers it sends forward changes. Follow what happens, Our neuron's input has changed, which changes its output, and that changes the loss and the gradients that flow back through the network.

As a result, the weights in our layer may take a different-sized step from one update to the next.

Each layer's weights are tuned for inputs of a certain size, but that size keeps changing. Each layer ends up chasing a moving target. The loss jumps around instead of sliding down, forcing us to use a smaller learning rate so those jumps don't throw training off course.

![covariate shift](fig/ch3-covariate-shift.svg)

A layer's input keeps changing because the layers before it keep changing during training. This drift is called **Internal Covariate Shift**, a term introduced in the 2015 paper [Batch Normalization](https://arxiv.org/abs/1502.03167).

Researchers still debate how much of the story this drift explains. A later study [(How Does Batch Normalization Help Optimization?)](https://arxiv.org/abs/1805.11604) found that normalization still improves training even when this drift is deliberately reintroduces. Its authors argued that the main benefit is a smoother loss: the loss changes more gently as the weights move, so the optimizer can take larger steps without destabilizing training. Both explanations point to the same practical remedy: keep the numbers entering each layer at a relatively steady scale.

### Twelve layers make it worse

A drift inside one layer is bad. A stack of layers multiplies it.

Here's why. Think about how spread out the numbers in a vector are. Every layer changes that spread by some factor, perhaps a little more than 1 or a little less, depending on its weights. Nothing forces that factor to be exactly 1, and training keeps changing it.

Now stack twelve layers. A factor of $1.2$ per layer becomes

$$1.2^{12} \approx 8.9$$

and a factor of $0.8$ per layer becomes

$$0.8^{12} \approx 0.069$$

A 20 percent change per layer sounds harmless, yet after twelve layers it produces numbers almost 9 times larger, or about 15 times smaller. The real model we load later has 27 layers. There, the same factors give about $137$ and about $0.0024$.

This is a deliberately simplified picture: a real transformer layer does much more than multiply the spread by a fixed number. The example isolates one failure mode so we can see why controlling scale matters.

Let's watch this happen. We build two stacks of twelve linear layers and send a random sequence of 196 tokens through each. In the first stack, every layer multiplies the spread by about 0.8. In the second, every layer multiplies it by about 1.2. Function`std()` measures the spread.

(The weights are random numbers, and `nn.init.normal_` controls their size by giving them a standard deviation of $\text{gain} / \sqrt{768}$. Each output is a sum of 768 input-weight products. With weights at this scale, the output's standard deviation is roughly `gain` times the input's standard deviation. So `gain` acts like a knob that makes a layer shrink or grow its input.)

```python
import torch
import torch.nn as nn

torch.manual_seed(0)
config = VisionConfig()
D = config.hidden_size

def make_stack(gain):
    layers = []
    for _ in range(config.num_hidden_layers):
        layer = nn.Linear(D, D, bias=False)
        nn.init.normal_(layer.weight, std=gain / D ** 0.5)   # each layer multiplies the spread by about gain
        layers.append(layer)
    return layers

x = torch.randn(1, 196, D)                                  # [1, 196, 768], spread about 1
for gain in (0.8, 1.2):
    h = x
    for layer in make_stack(gain):
        h = layer(h)
    print(f"gain {gain}: spread after {config.num_hidden_layers} layers = {h.std().item():.4f}")
```

**Output:**

```text
gain 0.8: spread after 12 layers = 0.0694
gain 1.2: spread after 12 layers = 8.8254
```

As we can see, after twelve layers one stack whispers and the other shouts.

But there is a problem. A layer in a real network behaves somewhere between these two extremes, and training keeps moving it. So the next layer can never know exactly what size of numbers is coming in. To fix this, we force the numbers entering every layer to be of a fixed size, no matter what the earlier layers did and to do that, we first need a way to measure "size".

### Mean and spread

For our purposes, these two numbers (mean and spread) are enough, where its center is, and how spread out it is. The Greek letters may look fancy, but they are just names for two things you already know, an average and a distance from that average.

The **mean** is the center of the list, the plain average. For a vector of $D$ numbers

$$\mu = \frac{1}{D}\sum_{i=1}^{D} x_i$$

where $\mu$ (mu) is the mean and $x_i$ is the $i$-th number of the vector.

The **variance** measures how wide the list is. It is the average squared distance from the mean

$$\sigma^2 = \frac{1}{D}\sum_{i=1}^{D} \left(x_i - \mu\right)^2$$

where $\sigma^2$ (sigma squared) is the variance. Squaring makes every distance positive, so both the numbers below and above the mean count. The square root of the variance, $\sigma$, is the **standard deviation**. It brings the result back to the units of the numbers themselves. The standard deviation is the spread that `std()` measured earlier.

Normalizing takes two steps. First, subtract the mean from every number so that the new mean is 0. Then divide every number by the standard deviation, so the new spread is 1

$$\hat{x}_i = \frac{x_i - \mu}{\sigma}$$

here $\hat{x}_i$ (x hat) is the $i$-th normalized number.

Try it on $\mathbf{x} = [2, 4, 6, 8]$. Find the mean:

$$\mu = \frac{2 + 4 + 6 + 8}{4} = 5$$

Subtract it from every number and you get $[-3, -1, 1, 3]$. The squared distances are $9, 1, 1, 9$, so the variance and the standard deviation are

$$\sigma^2 = \frac{9 + 1 + 1 + 9}{4} = 5$$

$$\sigma = \sqrt{5} \approx 2.236$$

Divide by the standard deviation gives

$$\hat{\mathbf{x}} \approx [-1.34, -0.45, 0.45, 1.34]$$

Check it. The four numbers add up to 0, so the mean is 0. Their squares are exactly $\frac{9}{5}, \frac{1}{5}, \frac{1}{5}, \frac{9}{5}$, and their average is

$$\frac{20}{5} \cdot \frac{1}{4} = 1$$

so the spread is 1.

Now try $[20, 40, 60, 80]$, the same vector scaled up by 10. The mean is 50 and the distances from the mean are $[-30, -10, 10, 30]$, that gives

$$\sigma^2 = \frac{900 + 100 + 100 + 900}{4} = 500$$

$$\sigma = \sqrt{500} \approx 22.36$$

Divide by the standard deviation, and you get exactly the same normalized values as before: $[-1.34, -0.45, 0.45, 1.34]$. For $[102, 104, 106, 108]$, the first vector moved up by 100. The mean is 105, the distances are again $[-3, -1, 1, 3]$, and the result is again the same.

Normalization removes two things: the overall size of the numbers and their shared offset. What it keeps is their relative pattern: which numbers are larger or smaller than the others, and how far apart they are relative to the overall spread.

![normalization](fig/ch3-normalization.svg)

In a batch of $B$ images, each with 196 tokens of 768 numbers, which numbers should we average over? There are two choices. We can take one feature and average it across the images in the batch, or we can take one token and average across its own 768 numbers. The first choice is the older one.

### Batch normalization

The first widely used answer was **Batch Normalization** ([Ioffe and Szegedy, 2015](https://arxiv.org/abs/1502.03167)), usually called batch norm. It was built for CNNs. Picture the batch as a table with one row per example and one column per feature. Batch norm works down each column: for each feature, it computes the mean and variance across all the examples in the batch:

$$\mu_f = \frac{1}{B}\sum_{b=1}^{B} x_{b,f} \qquad \sigma_f^2 = \frac{1}{B}\sum_{b=1}^{B} \left(x_{b,f} - \mu_f\right)^2$$

where $x_{b,f}$ is feature $f$ of example $b$, and $\mu_f$ and $\sigma_f^2$ are the mean and variance of feature $f$ across the $B$ examples.

It works very well for CNNs trained with large batches. But there is a catch: the normalized value of one example depends on whatever else happens to be in its batch.

Here is a small example. Put our dog photo in a batch of two images, and suppose its first feature is 2. If the other image has a 4 in that feature, the mean is 3, the distances are $-1$ and $1$, and the variance is 1, the dog's feature becomes:

$$\frac{2 - 3}{1} = -1$$

Now change the other image's feature to 0, the mean becomes 1, the distances are $1$ and $-1$, and the variance is still 1. This time the dog's feature becomes:

$$\frac{2 - 1}{1} = 1$$

Same dog, same number, opposite sign, only because the other image changed.

Real batches are larger, so the effect is milder, but it never completely disappears. The batch statistics also become noisier as the batch gets smaller. At inference, `BatchNorm` normally uses running averages collected during training rather than statistics computed from the current batch. The layer therefore behaves differently during training and inference.

Text(inputs) makes this even worse. Sentences have different lengths, so shorter ones filled with padding, extra tokens added to make every sentence in their batch the same length. Those padding tokens can affect the batch statistics.

What if we normalized each token using its own numbers?

### Layer normalization

**Layer normalization** ([Ba, Kiros and Hinton, 2016](https://arxiv.org/abs/1607.06450)), or layer norm, fixes this by switching direction.

Instead of working across the batch, it works across the features of each individual token. Each token gets its own mean and variance, computed from its own $D$ numbers. These are exactly the $\mu$ and $\sigma^2$ from the previous section on mean and spread. Nothing else in the batch is involved.

In our tensor of shape $[B, 196, 768]$, that means each of the $B \times 196$ token vectors is normalized independently, using its 768 features. Those features lie along the last axis, so in code the mean is taken with `dim=-1`.

A patch of sky and a patch of fur are each normalized using their own statistics. The result for one image does not change just because we put it in a batch of 1 or a batch of 1000. 

![batch norm vs layer norm](fig/ch3-batchnorm-vs-layernorm.svg)

This is why Transformers almost always use layer norm, or a close cousin of it, rather than batch norm.

### The learned scale and shift

But forcing every token to mean 0 and spread 1 could erase something useful. Maybe a layer works best when some of its numbers are larger than others, or when the whole representation is shifted away from zero.

Normalization should steady the numbers, not decide them for the model.

To fix this, layer norm ends with a learned scale and a learned shift

$$y_i = \gamma_i , \hat{x}_i + \beta_i$$

where $\gamma_i$ (gamma) is the learned scale for feature $i$ and $\beta_i$ (beta) is the learned shift. There is one $\gamma_i$ and one $\beta_i$ for each of the $D$ features, so each is a vector of $D$ numbers, shared by every token. They start at $\gamma = 1$ and $\beta = 0$, so at first layer norm is pure normalization. Training then adjusts them to whatever values work best.

Take our normalized $[-1.34, -0.45, 0.45, 1.34]$ and use $\gamma = 2$ and $\beta = 1$ for every feature. Each number is doubled and then shifted upward by 1, giving approximately 
$$[-1.68, 0.11, 1.89, 3.68]$$

The mean is now 1 and the spread is 2.

Doesn't that bring the drift back? No. The important difference is that $\gamma$ and $\beta$ belong to the normalization layer itself. They are the same for every token and every input, and they change only through training. The earlier layers can produce representations at different scales, but normalization removes the input-dependent scale before applying the learned feature-wise scale and shift.
### Guarding against zero

One danger is left in the formula. We divide by $\sigma$. What if all the numbers in a token are equal, say $[5, 5, 5, 5]$? The mean is 5 and every distance from the mean is 0, so $\sigma = 0$, and we compute $\frac{0}{0}$. That gives NaN, and once NaN enters a computation it breaks everything that depends on it.

To prevent this, we add a tiny number $\epsilon$ (epsilon) to the variance before the square root. The full recipe of layer norm is

$$\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} \qquad y_i = \gamma_i , \hat{x}_i + \beta_i$$

where $\epsilon$ is `layer_norm_eps=1e-6` from our `VisionConfig`, that is, 0.000001.

With $\epsilon$, the division for $[5, 5, 5, 5]$ is safe

$$\frac{0}{\sqrt{0.000001}} = \frac{0}{0.001} = 0$$

For an ordinary token, like $[2, 4, 6, 8]$ with $\sigma^2 = 5$, adding 0.000001 changes nothing you would notice.

### Layer norm by hand

Here is the whole recipe in a few lines of code, we apply it to $[2, 4, 6, 8]$ and to the same vector scaled up by 10, with one token per row.

We compute the variance ourselves as the average of the squared distances from the mean. PyTorch's `var()` divides by $D - 1$ by default instead of $D$, a correction used when estimating a variance from a small sample. Layer norm divides by $D$. 

On 4 numbers the difference is large: $\frac{20}{3} \approx 6.67$ instead of $5$.


```python
def layer_norm(x, gamma, beta, eps):
    mean = x.mean(dim=-1, keepdim=True)                     # [2, 1], one mean per token
    var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)      # [2, 1], divides by D, not D - 1
    x_hat = (x - mean) / torch.sqrt(var + eps)              # mean 0, spread 1
    return gamma * x_hat + beta

x = torch.tensor([[2., 4., 6., 8.],
                  [20., 40., 60., 80.]])                    # [2, 4], two tokens of 4 numbers
gamma, beta = torch.ones(4), torch.zeros(4)                 # the starting values
print(layer_norm(x, gamma, beta, eps=config.layer_norm_eps))

ln = nn.LayerNorm(4, eps=config.layer_norm_eps)
print(ln(x).detach())

flat = torch.tensor([[5., 5., 5., 5.]])                     # [1, 4], a token whose numbers are all equal
print(layer_norm(flat, gamma, beta, eps=config.layer_norm_eps))
print(layer_norm(flat, gamma, beta, eps=0.0))               # no guard
```

**Output:**

```text
tensor([[-1.3416, -0.4472, 0.4472, 1.3416], [-1.3416, -0.4472, 0.4472, 1.3416]]) tensor([[-1.3416, -0.4472, 0.4472, 1.3416], [-1.3416, -0.4472, 0.4472, 1.3416]]) tensor([[0., 0., 0., 0.]]) tensor([[nan, nan, nan, nan]])
```

The first two results show the values we worked out by hand, about $-1.34, -0.45, 0.45, 1.34$, and our function agrees with PyTorch's. The flat token gives four clean zeros with $\epsilon$, without the guard, it produces four NaNs.

You write this function once to understand what is happening. In the model we simply use `nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)`, which performs the same computation and stores $\gamma$ and $\beta$ for us. It stores $\gamma$ as `weight` and $\beta$ as `bias`, and those are the names the pretrained checkpoint uses too.

Always pass `eps` explicitly. `nn.LayerNorm` defaults to `1e-5`, but the pretrained encoder was trained with `1e-6`. Leaving it out will not produce an error, the model will still load and run, but it will compute slightly different numbers from the ones used during training.

### Layer norm on our patches
Now apply it to the patch vectors produced by `VisionEmbeddings`, our module from the last chapter. We'll make one random image and a second copy with every pixel value multiplied by 10, Then we'll send both through `VisionEmbeddings`, and compare the spread of the first token before and after the norm.

```python
torch.manual_seed(0)
config = VisionConfig()
embeddings = VisionEmbeddings(config)
norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

image = torch.randn(1, 3, 224, 224)                         # one random "image"
x = torch.cat([image, image * 10])                          # [2, 3, 224, 224], the image, then every pixel times 10
with torch.no_grad():
    h = embeddings(x)                                       # [2, 196, 768]
    out = norm(h)                                           # [2, 196, 768], every token normalized on its own

print(out.shape)
print("spread of token 0, before:", h[:, 0].std(dim=-1, unbiased=False))     # [2], one value per image
print("spread of token 0, after: ", out[:, 0].std(dim=-1, unbiased=False))   # [2]
print("mean of token 0, after:   ", out[:, 0].mean(dim=-1))                  # [2]
print("gamma", tuple(norm.weight.shape), "beta", tuple(norm.bias.shape))
```

**Output:**

```text
torch.Size([2, 196, 768])
spread of token 0, before: tensor([1.1104, 5.7675])
spread of token 0, after: tensor([1.0000, 1.0000])
mean of token 0, after: tensor([-4.9671e-09, 3.5701e-09])
gamma (768,) beta (768,)
```

The shape does not change, $[2, 196, 768]$ goes in and $[2, 196, 768]$ comes out. Layer norm preserves the Transformer's contract: $N$ vectors in, $N$ vectors of the same length out.

Before the norm, the second image's token has a much larger spread than the first: 5.7675 versus 1.1104. After the norm, both spreads are 1 up to rounding, $[1.0000, 1.0000]$. The means are tiny numbers such as $10^{-9}$ rather than an exact 0, because computers round their calculations.

Notice that the two normalized tokens are not identical. Earlier, $[2, 4, 6, 8]$ and $[20, 40, 60, 80]$ normalized to exactly the same numbers. Here the second token is not exactly 10 times the first. Multiplying the pixels by 10 multiplies the patch projection by 10, but the projection's bias and the position vector are added afterwards, unchanged. Normalization still brings both tokens the same overall scale, but it does not guarantee that their normalized patterns will be identical.

How big is a norm? $\gamma$ and $\beta$ each hold 768 numbers, so one norm has $2 \times 768 = 1{,}536$ parameters. Every encoder layer has two norms, and one final norm follows the stack. For our config that gives

$$2 \times 12 + 1 = 25 \text{ norms}$$

$$25 \times 1{,}536 = 38{,}400 \text{ parameters}$$

The real model has $2 \times 1{,}152 = 2{,}304$ parameters per norm and 27 layers, so

$$2 \times 27 + 1 = 55 \text{ norms}$$

$$55 \times 2{,}304 = 126{,}720 \text{ parameters}$$

That is tiny next to the millions of weights in attention and the MLP.

### Twelve layers, steady again

Now go back to the two stacks from the start of the chapter, and put a layer norm in front of every layer.

```python
torch.manual_seed(0)
x = torch.randn(1, 196, D)                                  # [1, 196, 768]
for gain in (0.8, 1.2):
    h = x
    for layer in make_stack(gain):
        norm = nn.LayerNorm(D, eps=config.layer_norm_eps)   # gamma 1, beta 0, pure normalization
        h = layer(norm(h))                                  # the norm resets the spread to 1 before every layer
    print(f"gain {gain}: spread after {config.num_hidden_layers} layers = {h.std().item():.4f}")
```

**Output:**

```text
gain 0.8: spread after 12 layers = 0.7990
gain 1.2: spread after 12 layers = 1.1975
```

The first stack now ends at 0.7990 and the second at 1.1975, which is close to the factor of a single layer. The last layer still changes the spread by its own factor, but it receives a normalized input each time. Nothing from the earlier layers pile up. Instead of multiplying twelve factors together, we only see the effect of the final layer.

### What the first layer really receives

So far our norms have only seen numbers we made up. Time for the real thing. The ViT we loaded in the last chapter is fully trained, and the first thing its first encoder layer does is a layer norm.

Let's give it the dog image from the last chapter.

First the photo has to be converted into the numbers this model expects. Dividing by 255 puts every pixel value between 0 and 1. Subtracting 0.5 and dividing by 0.5 then puts the values between −1 and 1. This is a normalization too, but there is an important difference. They are fixed numbers (0.5,0.5), same for every image. Layer norm, by contrast, computes its statistics separately from each token. The processor we build later will handle this step for us. Our real model uses the same normalization.

Then we take the `VisionEmbeddings` written in the last chapter and copy the trained patch weights and position vectors into it. We compare three versions of the same 196 vectors: the vectors as they leave the embeddings, the vectors after a plain norm with $\gamma = 1$ and $\beta = 0$, and the vectors after the trained norm with its learned $\gamma$ and $\beta$.

This model uses $\epsilon = 10^{-12}$ instead of $10^{-6}$.

To find the trained norm, the code looks for the first layer norm inside the model and prints its name. In our own encoder the corresponding norm will be called `layer_norm1`.

Hugging Face also adds a CLS token, so its first norm sees 197 tokens rather than 196. We run the norm on all 197 tokens too, drop the CLS token, and check that our 196 patch vectors are unchanged. This also gives us a useful test of the code from the last chapter. Hugging Face builds its patch vectors with its own implementation, while you built yours with `VisionEmbeddings`. If your patch cutting, flattening, transpose or position lookup is wrong anywhere, the two implementations will not match and the checks below will print `False`.

```python
# VisionConfig and VisionEmbeddings are the classes you wrote in the last chapter.
# If they are wrong, the "same with or without CLS" check below prints False.
import numpy as np
from PIL import Image
from transformers import ViTModel

vit = ViTModel.from_pretrained("google/vit-base-patch16-224", add_pooling_layer=False).eval()
config = VisionConfig()                # same sizes as this checkpoint
emb = VisionEmbeddings(config).eval()  # your module, with the trained weights copied in
with torch.no_grad():
    emb.patch_embedding.weight.copy_(vit.embeddings.patch_embeddings.projection.weight)# [768, 3, 16, 16]
    emb.patch_embedding.bias.copy_(vit.embeddings.patch_embeddings.projection.bias)    # [768]
    emb.position_embedding.weight.copy_(vit.embeddings.position_embeddings[0, 1:])        # [197, 768] -> [196, 768], CLS row dropped

photo = np.array(Image.open("dog.jpg").convert("RGB").resize((224, 224)))                 # [224, 224, 3]
pix = torch.tensor(photo).permute(2, 0, 1).float().div(255).sub(0.5).div(0.5)[None]      # [1, 3, 224, 224], values in [-1, 1]

norms = [(name, m) for name, m in vit.named_modules() if isinstance(m, nn.LayerNorm)]
ln_name, ln_real = norms[0]     # the first norm of the first encoder layer
print("first norm:", ln_name)
ln_plain = nn.LayerNorm(config.hidden_size, eps=vit.config.layer_norm_eps).eval()   # gamma 1, beta 0

with torch.no_grad():
    patches = emb(pix)[0]            # [1, 196, 768] -> [196, 768]
    plain = ln_plain(patches)        # [196, 768], every patch at mean 0, spread 1
    trained = ln_real(patches)       # [196, 768], then the trained scale and shift
    full = ln_real(vit.embeddings(pix))[0, 1:] 
    # [1, 197, 768] -> [196, 768], CLS normalized too, then dropped

print("same with or without CLS:", torch.allclose(trained, full, atol=1e-5))
print(f"gamma from {ln_real.weight.min():.2f} to {ln_real.weight.max():.2f}, "
      f"beta from {ln_real.bias.min():.2f} to {ln_real.bias.max():.2f}")

side = config.image_size // config.patch_size       # 14 patches per side
stages = {"before the norm": patches, "normalized": plain, "after gamma and beta": trained}
spreads = {t: m.std(dim=-1, unbiased=False).view(side, side) for t, m in stages.items()}   # [196] -> [14, 14] each
for t, s in spreads.items():
    print(f"{t:22s} spread from {s.min():.3f} to {s.max():.3f}")

vmax = max(s.max() for s in spreads.values()).item()  # one color scale for all three maps
fig, ax = plt.subplots(1, 4, figsize=(16, 4))
ax[0].imshow(photo); ax[0].set_title("image")
for a, (t, s) in zip(ax[1:], spreads.items()):
    im = a.imshow(s.numpy(), cmap="Oranges", vmin=0, vmax=vmax); a.set_title(t)
fig.colorbar(im, ax=ax[1:].tolist(), shrink=0.8)
for a in ax: a.axis("off")
plt.show()
```

**Output:**

```text
first norm: layers.0.layernorm_before
same with or without CLS: True
gamma from 0.03 to 0.23, beta from -0.14 to 0.20
before the norm          spread from 0.394 to 1.351
normalized               spread from 1.000 to 1.000
after gamma and beta     spread from 0.094 to 0.148
```

![norm-output](fig/ch3-norm-output.png)

Each map shows the spread of one patch at the position where the patch came form in the picture. Let's read them one at a time.

**Before the norm.** The spread varies from patch to patch, from 0.394 to 1.351. The widest patch is about $1.351 / 0.394 \approx 3.4$ times wider than the narrowest. You will not find a clean outline of the dog in this map. The last chapter explains why: each vector is the patch's content plus its position vector, so its size reflects both what the patch contains and where that patch sits. This is exactly what the first layer would receive without a norm: numbers whose scale depends on the photo, the patch and its position. A different photo could produce a different pattern of scales, leaving the layer to deal with whatever happens to arrive.

**After the plain norm.** Every patch has spread exactly 1.000, so the map becomes one flat color. The scale information is while the relative patterns inside each vector remain. Whatever the photo does to the magnitude of a patch, the first layer now receives every patch at the same normalized scale.

**After the trained $\gamma$ and $\beta$.** The spread now ranges from 0.094 and 0.148. Training did not choose to keep spread 1. Instead the learned $\gamma$ values substantially shrink the normalized features. The exact spread of each patch depends on both its normalized values and the feature-wise $\gamma$ values applied to them. Attention multiplies these values by its own weights next, so the model is free to choose one scale here and compensate for it elsewhere.

The spreads are no longer exactly equal: they range from 0.094 to 0.148. Each feature has its own $\gamma$ and $\beta$, so different patches can end up with slightly different spreads depending on which feature values they contain. But compare the ranges. Before the norm, the widest patch was about 3.4 times wider than the narrowest. After it, the ratio is about $0.148 / 0.094 \approx 1.6$.

And there is one difference that matters more than the ranges. Before the norm, the scale of each patch was determined by the incoming representation. After normalization, the input-dependent scale has been removed, and the learned $\gamma$ and $\beta$ provide a stable, feature-wise transformation that is shared across inputs.

Show this model a dark photo, a bright photo or a photo of something completely different. The plain normalized representation will still have unit spread for every patch, while the learned norm will apply the same trained parameters to every image.

The second line of the output reads `True`. Hugging Face normalized 197 tokens, including CLS, while we normalized only the 196 patch tokens, yet the patches came out the same.

Why? Because each token is normalized using only its own 768 features. Removing the CLS token therefore cannot change the normalization of any patch token. With batch norm, other examples would affect the statistics. With layer norm, they do not.

**That is the main idea of this chapter**, now seen on a real photo through a real trained model: before each sublayer, each token is normalized using only its own features, so the scale of the representation no longer depends on whatever magnitude happened to come from the layers before it.

### Where the norms sit

This is where the 25 norms go in our encoder, using the names from the checkpoint of the real model we load later on. The layers are numbered from 0, the way the code counts them, just like the `layers.0` you saw in the output above.

```text
  patch vectors      [B, 196, 768]
  layer 0            layer_norm1 -> attention -> layer_norm2 -> MLP
  layer 1            layer_norm1 -> attention -> layer_norm2 -> MLP
  ...
  layer 11           layer_norm1 -> attention -> layer_norm2 -> MLP
  post_layernorm     [B, 196, 768]
```

(The two shortcuts inside every layer are left out of this picture. We will add them when we build the layer.)

Each norm sits in front of the sublayer. Attention and the MLP are two main pieces inside an encoder layer, and are often called sublayers. Both receive normalized inputs.

This arrangement, norm first and the sublayer after, is called **pre norm**. The ViT implementation uses it, and that is why Hugging Face named its first norm `layernorm_before`.

The original Transformer paper used the opposite arrangement, called **post norm**: the sublayer output was added to the residual stream and the result was then normalized. Pre norm became popular because it made the deep Transformer stack easier to optimize ([Xiong et al., 2020](https://arxiv.org/abs/2002.04745)).

However, Pre norm leaves one gap. The norms inside the layers normalize the inputs to attention and the MLP, but the main stream of vectors travels from layer to layer is not itself normalized.
This will make more sense once the shortcuts (residual connections) are introduced.

There is one last type of norm, `post_layernorm`, which normalizes the 196 vectors on their way out of the encoder.

The language model also normalizes before its sublayers, but with a slimmer cousin of layer norm called **RMSNorm**. It skips subtracting the mean and keeps only the rescaling, with its own $\epsilon$. We will build it with the decoder.

Notice what the norm does not do. Like the patch embedding, it works on each token independently. It reads one token's 768 features and nothing else. This is exactly why the CLS token could be dropped without changing any of the patch tokens. After the norm, the ear still knows nothing about the snout. The norm only makes sure that, when the patches finally start to talk, they all speak at the same volume. The talking itself is attention's job.

![ch3-so-far](fig/ch3-so-far.svg)

We now have the piece that keeps every layer's input to the sublayers steady. But steady inputs are not enough to make a deep stack train well. Every layer still transforms the representation, and on the way back the gradient has to pass through those transformations to reach the earlier layers. The deeper the stack, the more important it becomes to give the signal a direct path through the network. In the next chapter we will build the encoder layer around the norms, fill in the MLP, and add the two shortcuts that give the signal a direct road through all twelve layers.

---

# Chapter 4. Stacking Layers without Losing the Signal

At the end of the last chapter we had a norm in front of every sublayer, so each sublayer now receives its input at a steady scale. But steady inputs do not solve everything. Every layer still transforms its input, so what comes out of the last layer is a copy of a copy of a copy. Twelve times over.

On the way back, the gradient has to pass through every one of those transformations before it reaches the early layers. In this chapter we fix this problem in three steps: **we build the MLP, we add the two shortcuts, and then we put everything together into a full encoder**.

> **Main Idea: Every encoder layer keeps its input and only adds corrections to it, so the signal and the gradient both have a direct road through the whole stack.**

Think of the twelve layers as twelve editors working on one page. Without shortcuts, each editor reads the page, throws it away and writes a new one from memory. With shortcuts, the original page is passed along, and each editor only adds notes in the margin.

### What one layer does

You saw the four steps of a layer in the **ViT figure** earlier: a norm, attention, another norm and the MLP. Now we add a shortcut around each sublayer. Each shortcut goes around a norm and its sublayer, then adds the original input back to the result.

![one encoder layer](fig/ch4-encoder-layer.svg)

The layer does two things in order: It normalizes its input, runs attention on it, and adds the result back to the input. After that's done, it normalizes the new sum, runs the MLP on it, and adds the result back to the sum.

The same two steps, written as symbols:

$$\mathbf{h} = \mathbf{x} + \text{Attention}\big(\text{LN}_1(\mathbf{x})\big)$$

$$\mathbf{y} = \mathbf{h} + \text{MLP}\big(\text{LN}_2(\mathbf{h})\big)$$

here, $\mathbf{x}$ is one token's vector as it enters the layer, $\text{LN}_1$ and $\text{LN}_2$ are the two layer norms, $\mathbf{h}$ is the vector halfway through, and $\mathbf{y}$ is what the layer passes on. All three vectors have 768 numbers, so they can be added.

Here is the same idea with one number instead of 768. A token enters with $x = 2$, attention adds $0.5$ and the MLP adds $-0.3$:

$$h = 2 + 0.5 = 2.5$$

$$y = 2.5 - 0.3 = 2.2$$

The layer did not replace the 2. It nudges it.

### Who mixes and who works alone

The two sublayers do very different jobs.

**Attention** is the only part of the layer where tokens share information. It lets the ear patch look at the snout patch and pull in what it needs.

**The MLP** works on each token independently. Patch 5 goes through the MLP without ever seeing patch 90. The same MLP, with the same weights, is applied to each of the 196 vectors one at a time. MLP digests what attention gathered. That is its job: attention brings information into a token, and the MLP works with that information inside the token.

So far, everything we have built works on one token at a time: the patch embedding, the position vector, the norm, and now the MLP. Attention is the only exception.

### The MLP: expand, bend, compress

The MLP does three things to each token:

**Expand.** A linear layer called `fc1` turns the 768 numbers into 3072 numbers.
**Bend.** A function called GELU bends each of those numbers.
**Compress.** A second linear layer called `fc2` turns the 3072 numbers back into 768.

("fc" stands for fully connected, an older name for a linear layer. You will also see the MLP called the **feed forward network**.)

$$\text{MLP}(\mathbf{x}) = W_2 \thinspace \text{GELU}\big(W_1\mathbf{x} + \mathbf{b}_1\big) + \mathbf{b}_2$$

here, $W_1$ and $\mathbf{b}_1$ are the weights and biases of `fc1`, $W_2$ and $\mathbf{b}_2$ of `fc2`. 
$W_1$ has shape $[3072, 768]$ and $W_2$ has shape $[768, 3072]$. (PyTorch writes a linear layer's weight shape as output size first, then input size.)

**Why make the vector bigger in the middle?** It gives the MLP more room to work. You can think of each of the 3072 middle numbers as a small detector that checks the token for one pattern. GELU decides how strongly each detector's answer counts, and `fc2` combines all 3072 answers back into 768 numbers.

Making the middle four times wider is the usual choice: 768 and 3072 in our config, 512 and 2048 in the original Transformer. The real model we load later uses 1152 and 4304. This size was chosen by a study that searched for the best shape for a given compute budget ([Getting ViT in Shape, 2023](https://arxiv.org/abs/2305.13035)).

![expand and compress](fig/ch4-mlp-expand-compress.svg)

### Why the bend matters

Why not skip the bend and keep just the two linear layers? Because two linear layers in a row are no good than a single linear layer.

Let's work an example for this. Say, first layer computes $y = 2x + 1$ and the second computes $z = 3y - 4$. Substituting the equations, we get:

$$z = 3(2x + 1) - 4 = 6x - 1$$

That is just one linear layer, with weight 6 and bias $-1$. At $x = 1$. The two layers give $y = 3$ and then $z = 3 \cdot 3 - 4 = 5$, and the single layer gives $6 \cdot 1 - 1 = 5$. Same answer.

The same thing happens with matrices of any size:

$$W_2\big(W_1\mathbf{x} + \mathbf{b}_1\big) + \mathbf{b}_2 = \big(W_2W_1\big)\mathbf{x} + \big(W_2\mathbf{b}_1 + \mathbf{b}_2\big)$$

The right side is a single matrix $W_2W_1$ of shape $[768, 768]$ plus a single bias. So without a bend, `fc1` and `fc2`, with their 4.7 million weights, can do nothing that one $768 \times 768$ matrix could not. Even a thousand linear layers in a row would still behave like one.

### ReLU and GELU

A function that bends the numbers between two linear layers is called an **activation function**.

The simplest one is **ReLU**, short for Rectified Linear Unit. It keeps positive numbers as they are and turns negative numbers into 0:

$$\text{ReLU}(x) = \max(0, x)$$

So $\text{ReLU}(3) = 3$ and $\text{ReLU}(-3) = 0$.

Put ReLU between our two small layers and they stop collapsing. At $x = 1$, the first layer gives 3, ReLU leaves it unchanged, and the second layer gives 5, the same as the line $6x - 1$. But at $x = -1$, the first layer gives $-1$, ReLU turns it into 0, and the second layer gives $3 \cdot 0 - 4 = -4$. The line $6x - 1$ would have given $-7$. The two now disagree, so the network is no longer a single straight line.

ReLU however has one weakness. For every negative input, its output is exactly 0, and so its gradient also becomes 0. A neuron that receives a negative number learns nothing from that token.

Our model uses a softer bend called **GELU**, short for Gaussian Error Linear Unit ([Hendrycks and Gimpel, 2016](https://arxiv.org/abs/1606.08415)). BERT, GPT-2 and the ViT all use it. GELU behaves like ReLU for large numbers, but bends smoothly near 0 instead of cutting off sharply:

$$\text{GELU}(x) = x \cdot \Phi(x)$$

here, $\Phi(x)$ (capital phi) tells us how likely a random number from the bell curve is to be smaller than $x$. For large positive $x$,  $\Phi(x)$ is close to 1, so GELU leaves $x$ almost unchanged. For a large negative $x$,  $\Phi(x)$  is close to 0, so GELU pushes the output towards 0. Around 0, $\Phi(x)$ is about $0.5$, so GELU keeps roughly half of the input.

Here are a few values side by side:

| $x$  | $\Phi(x)$ | $\text{GELU}(x)$ | $\text{ReLU}(x)$ |     |
| ---- | --------- | ---------------- | ---------------- | --- |
| $-1$ | $0.1587$  | $-0.1587$        | $0$              |     |
| $0$  | $0.5$     | $0$              | $0$              |     |
| $1$  | $0.8413$  | $0.8413$         | $1$              |     |
| $2$  | $0.9772$  | $1.9545$         | $2$              |     |

For values far from 0, GELU and ReLU behave similarly. Around 0 however, GELU is gentler: a small negative input produces a small negative output instead of being cut off at 0. GELU reaches a minimum of about $-0.17$, near $x = -0.75$.

There is one practical detail. $\Phi$ does not have a short expression that is convenient to compute, so many models use a $\tanh$ based approximation that is almost identical to GELU. The real model was trained with this tanh version. Its config specifies `gelu_pytorch_tanh`, and PyTorch gives us that version with `F.gelu(x, approximate="tanh")`. We therefore use the same curve the model was trained with.

If you are curious, the approximation is:
$$0.5 x \left(1 + \tanh\left(\sqrt{2/\pi} \big(x + 0.044715 x^3\big)\right)\right)$$
here $\tanh$ squashes a number into the range $(-1, 1)$. You will never need to calculate this by hand. PyTorch handles it for us.

![GELU and ReLU](fig/ch4-gelu-vs-relu.svg)

Here is a complete MLP that is small enough to work through by hand: one number goes in, two numbers appear in the middle, and one number comes out. Let `fc1` have weights $[1, -1]$ and `fc2` have weights $[1, 1]$, with no biases. Feed it $x = 1$: `fc1` expands it to $[1, -1]$, GELU bends that to about $[0.841, -0.159]$ and `fc2` adds the two: $0.841 - 0.159 = 0.682$.

Without GELU, `fc2` would simply add $1$ and $-1$ and get 0, and the same cancellation would happen for every input. With GELU in the middle, the two values are changed before they are added, so the output depends on $x$ in a way no single linear layer can reproduce.

Let's check these numbers with PyTorch.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

x = torch.linspace(-4, 4, 1001)                      # 1001 points from -4 to 4
relu = F.relu(x)
gelu = F.gelu(x)                                     # exact, x times Phi(x)
gelu_tanh = F.gelu(x, approximate="tanh")            # the version our model uses

print(f"lowest GELU value: {gelu.min():.4f} at x = {x[gelu.argmin()]:.3f}")
print(f"largest gap between exact and tanh GELU: {(gelu - gelu_tanh).abs().max():.6f}")

plt.plot(x, relu, label="ReLU")
plt.plot(x, gelu_tanh, label="GELU (tanh)")
plt.axhline(0, color="grey", lw=0.5); plt.legend(); plt.show()
```

**Output:**

```text
lowest GELU value: -0.1700 at x = -0.752
largest gap between exact and tanh GELU: 0.000473
```

![relu-gelu-output](fig/ch4-relu-gelu.png)

The lowest GELU value is about $-0.17$ near $x = -0.75$, and the largest gap between the two GELU curves anywhere from $-4$ to $4$ is $0.000473$.

Now let's confirm, at full size, that two linear layers really do collapse into one. We build `fc1` and `fc2`, combine them by hand into a single matrix, and bias, and compare their outputs. Then we put GELU in between them and check the result again.

```python
torch.manual_seed(0)
config = VisionConfig()
fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
x = torch.randn(1, 196, config.hidden_size)                   # [1, 196, 768]

with torch.no_grad():
    two = fc2(fc1(x))                                         # [1, 196, 3072] -> [1, 196, 768]
    W = fc2.weight @ fc1.weight                               # [768, 3072] @ [3072, 768] -> [768, 768]
    b = fc2.weight @ fc1.bias + fc2.bias                      # [768]
    one = x @ W.T + b                                         # [1, 196, 768], a single linear layer
    bent = fc2(F.gelu(fc1(x), approximate="tanh"))            # [1, 196, 768], with the bend in the middle

print("two linear layers == one linear layer:", torch.allclose(two, one, atol=1e-4))
print("with GELU in between, still one layer? ", torch.allclose(bent, one, atol=1e-4))
```

**Output:**

```text
two linear layers == one linear layer: True
with GELU in between, still one layer? False
```

The first line is `True`: the two linear layers and the merged single layer produce the same 196 vectors. The second should reads `False`: once GELU sits in the middle, a single matrix can longer reproduce the result.

So the bend is what gives the MLP its extra expressive power. Without it, the second linear layer would be unable to add any new capability.

### The MLP module

Here is the module itself. Like the modules we built earlier, it gets its dimensions from the config.
`fc1` and `fc2` also match the names used by the pretrained checkpoint.

```python
class VisionMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states)                     # [B, N, 3072]
        hidden_states = F.gelu(hidden_states, approximate="tanh")
        hidden_states = self.fc2(hidden_states)                     # [B, N, 768]
        return hidden_states
```

The shapes show exactly what happens. `fc1` changes $[B, N, 768]$ into $[B, N, 3072]$ ($B$ is the batch size and $N$ is the number of tokens). GELU acts on each number independently, so the shape does not change. `fc2` then brings the last dimension back to 768, giving $[B, N, 768]$. Because a linear layer operates only on the last dimension, every token goes through the MLP independently.

**How many weights does the MLP have?** A linear layer has one weight for every input-output pair, plus one bias for each output:

$$\text{parameters} = d_{\text{in}} \cdot d_{\text{out}} + d_{\text{out}}$$

here, $d_{\text{in}}$ and $d_{\text{out}}$ are the input and output sized. A small layer that maps 2 numbers to 3 numbers has $2 \cdot 3 + 3 = 9$ parameters.

For our MLP:

| Layer | Calculation             | Parameters |
| ----- | ----------------------- | ---------- |
| `fc1` | $768 \cdot 3072 + 3072$ | 2,362,368  |
| `fc2` | $3072 \cdot 768 + 768$  | 2,360,064  |
| MLP   | sum of both             | 4,722,432  |

That is about 3000 times as many parameters as the layer norm from the previous chapter, which has only 1536. In the real model, one MLP holds 9,921,872 parameters, and the 27 MLPs together hold 267,890,544.

Let's run the module and check that each token really goes through it independently. We will change patch 5 while leaving every other patch untouched, and then see which outputs change.

```python
torch.manual_seed(0)
mlp = VisionMLP(config)
x = torch.randn(1, 196, config.hidden_size)                                     # [1, 196, 768]
with torch.no_grad():
    y = mlp(x)                                                                  # [1, 196, 768]
    x2 = x.clone()
    x2[0, 5] += 1.0                                                             # change patch 5 only
    y2 = mlp(x2)                                                                # [1, 196, 768]

print(y.shape)
print("fc1 parameters:", sum(p.numel() for p in mlp.fc1.parameters()))
print("fc2 parameters:", sum(p.numel() for p in mlp.fc2.parameters()))
print("MLP parameters:", sum(p.numel() for p in mlp.parameters()))
changed = (y2 - y).abs().amax(dim=-1)[0]                                        # [196], biggest change per patch
print("patches whose output changed:", (changed > 0).nonzero().flatten().tolist())
```

**Output:**

```text
torch.Size([1, 196, 768])
fc1 parameters: 2362368
fc2 parameters: 2360064
MLP parameters: 4722432
patches whose output changed: [5]
```

The output shape is $[1, 196, 768]$, preserving the Transformer contract. The three parameter counts match the table: 2,362,368, 2,360,064 and 4,722,432. And the last line should list only `[5]`.

We changed patch 5, and none of the other 195 outputs changed. The MLP therefore processes each patch independently, inside the MLP, the ear patch never sees the snout

### Residual connections

Now that we have the pieces of a layer, we can return to the problem we identified at the end of the last chapter.

In 2015, a team at Microsoft Research trained two plain CNNs on the same images, one with 20 layers and another with 56. You would expect the deeper network to perform at least as well, but it did worse. Not only on new images but even on the images it was trained on ([Kaiming et al,.](https://arxiv.org/abs/1512.03385)). This was surprising. The 56 layer network, could in principle, copy the 20 layer network and let its extra 36 layers simply pass their input through unchanged. But the training failed to find that solution. For a stack of layers that each transform its input, learning to do nothing turned out to be surprisingly difficult.

The solution was the **Residual Connection**, also called a skip connection. Instead of replacing its input, a block computes a correction and adds it to the input:

$$\mathbf{y} = \mathbf{x} + f(\mathbf{x})$$

here, $\mathbf{x}$ is what enters the block's input, $f(\mathbf{x})$ is the correction computed by the block (for us, a norm followed by attention or by the MLP), and $\mathbf{y}$ is the output. (The name comes from the fact that the block only learns the difference between its output and its input, that difference is called the residual.)

![residual-connection](fig/ch4-residual-connection.svg)

Suppose $x = 2$ and the block computes a correction of $0.1$, the output is $2.1$. If the correction is 0, the output remains 2. This make doing nothing easy to learn. The block only needs to produce zeros. With residual connections, the same team trained a network of 152 layers and won the ImageNet competition of 2015. Residual connections have since become a standard part of deep networks, including Transformers.

That explains the benefits in the forward direction. But shortcuts are just as important when the gradient travels back.

During training, the gradient passes through every layer, and each layer can scale or distort it. Without shortcuts, those effects multiply. Suppose two layers let through factors $0.1$ and $0.2$ let through $0.1 \cdot 0.2 = 0.02$ of the gradient (only 2%). If 12 layers each have a factor of 0.1 the gradient is multiplied by $0.1^{12}$, about one part in a trillion. The early layers would receive almost no useful gradient.

A residual connection changes this, because the input is added directly to the output, the gradient has a direct path through the layer. In our simple one-number example the factor becomes $1+f$.
For the same two layers: $(1 + 0.1)(1 + 0.2) = 1.32$

Expanding the product gives $1 + 0.1 + 0.2 + 0.02$. Each term represents a different path through the two layers, the $0.1$ goes through the first layer only, the $0.2$  goes through the second layer only, and the $0.02$ goes through both. No matter how small the layers' own factors become, the direct path with weight 1 remains.

Real layers operate on whole vectors, so their factors are matrices rather than single numbers. The idea is the same: the shortcut always gives the gradient a path along which it can pass without being transformed by the sublayer.

Every shortcut doubles the number of possible paths, because the signal can either pass through the sublayer or go around it. Our encoder has 24 shortcuts, two in each of its 12 layers, so there are $2^{24}$ possible paths, or about 16.8 million. One of those paths skips every sublayer entirely. It is the direct path from the input of the stack to its output. A study of residual networks found that during training most of the gradient travels along the shorter paths. This suggests that a deep residual network behave somewhat like many shallower networks working together ([Residual Networks Behave Like Ensembles of Relatively Shallow Networks](https://arxiv.org/abs/1605.06431)).

Our editor analogy gives us the same picture. Without shortcuts, each editor writes a new page from memory, so whatever the first editor wrote has to survive eleven rewrites. With shortcuts, the original page is passed along the entire line, while each editor adds notes in the margin. Feedback from the reader at the end can travel directly back along that original page instead of being passed through eleven separate rewrites.

A shortcut makes "change nothing" the easiest thing for a layer to learn, while also keeping a direct path open for the gradient.

### The encoder layer

Now we have everything to write the layer itself. There is however one problem: the layer needs attention, and we have not built attention yet. For now we will use a stand-in, a class with the same interface that returns zeros. 

Because of the shortcut, a sublayer that returns zeros have no effect: $\mathbf{x} + \mathbf{0} = \mathbf{x}$. The layer can therefore runs normally even though attention in not implemented yet, the MLP does the actual computation and the attention stand-in does nothing.

The stand-in also returns a second value, `None`. The real attention module will return attention weights in that position. We will use those weights later to see which patches attend to which others. When we replace the stand-in with the the real attention, we replace only this class, and the rest of this chapter keeps working.

The attribute names `self_attn`, `layer_norm1`, `mlp` and `layer_norm2` match the names in the pretrained checkpoint.

```python
class VisionAttention(nn.Module):
    # stand-in until the next chapter: returns zeros, so only the shortcut and the MLP act
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, hidden_states):
        return torch.zeros_like(hidden_states), None   # [B, 196, 768] and no attention weights yet


class VisionEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = VisionAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = VisionMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        residual = hidden_states             # [B, 196, 768], kept for shortcut 1
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states=hidden_states)
        hidden_states = residual + hidden_states  # shortcut 1, around attention

        residual = hidden_states             # [B, 196, 768], kept for shortcut 2
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states # shortcut 2, around the MLP
        return hidden_states                     # [B, 196, 768]
```

Read the `forward` method alongside the two equations from the beginning of the chapter. They describe exactly the same computation. `residual` keeps the original input while the norm and sublayer process a copy of it. The original is then added back to the sublayer's output. Notice that the norm only acts on the value entering the sublayer. The copy carried by the shortcut is never normalized.

Let's check that the layer behaves as expected. Because attention currently returns zero, the first half of the layer leaves its input unchanged. The final output should therefore be the original input plus the MLP's correction.

```python
torch.manual_seed(0)
layer = VisionEncoderLayer(config)
x = torch.randn(1, 196, config.hidden_size)           # [1, 196, 768]
with torch.no_grad():
    y = layer(x)                                      # [1, 196, 768]
    correction = layer.mlp(layer.layer_norm2(x))      # [1, 196, 768], what the MLP adds (attention adds 0 for now)

print(y.shape)
print("output == input + MLP correction:", torch.allclose(y, x + correction, atol=1e-5))
print(f"spread of input {x.std():.4f}, of the correction {correction.std():.4f}")
```

**Output:**

```text
torch.Size([1, 196, 768])
output == input + MLP correction: True
spread of input 0.9984, of the correction 0.1991
```

The shape remains $[1, 196, 768]$, and the second line reads `True`: the output is equal to input plus the MLP's correction. The last line compares the spread of the input, $0.9984$, with the spread of the correction, $0.1991$. The layer keeps the original signal and adds a correction to it.

### Twelve layers with and without shortcuts

Now let's see what the shortcuts do when we stack many layers together. To isolate the effect of the shortcut, we use a simpler block: a norm followed by an MLP, with the shortcut either enabled or disabled. We stack twelve of these blocks and ask two questions.

**Forward: does the input survive?** We compare the input of the with its output, using the cosine similarity from chapter 1. A value close to 1 means the output still points in the same direction as the input. A value close to 0 means the output has lost track of the input.

**Backward: does the gradient survive?** We send a known signal backward through the stack and measure how much of it reaches the input. We call this signal `pull`. The trick is to define the loss as the element-wise product of the output and `pull`, summed over all elements. That makes the gradient at the output exactly equal to `pull`. We can then compare the gradient that arrives at the input with the signal we originally sent.

Both experiments use the same random seed, so the two runs get the same weights, the same input and the same `pull`. Only the shortcut differs.

```python
class Block(nn.Module):
    # a norm and an MLP, with or without a shortcut around them
    def __init__(self, config, shortcut):
        super().__init__()
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = VisionMLP(config)
        self.shortcut = shortcut

    def forward(self, x):
        out = self.mlp(self.norm(x))     # [1, 196, 768]
        return x + out if self.shortcut else out


for shortcut in (False, True):
    torch.manual_seed(0)                 # same weighs, input and pull in both runs
    blocks = [Block(config, shortcut) for _ in range(config.num_hidden_layers)]
    x = torch.randn(1, 196, config.hidden_size, requires_grad=True) # [1, 196, 768]
    pull = torch.randn(1, 196, config.hidden_size) # [1, 196, 768], the gradient we send down from the top

    h = x
    for block in blocks:
        h = block(h)                      # [1, 196, 768]
    loss = (h * pull).sum()               # its gradient at the top is exactly pull
    loss.backward()

    kept = F.cosine_similarity(h.flatten(), x.flatten(), dim=0).item() # forward: how much of the input survives
    arrived = F.cosine_similarity(x.grad.flatten(), pull.flatten(), dim=0).item()   # backward: how much of the pull survives
    print(f"shortcut {str(shortcut):5s}: input kept {kept:.3f} | pull arrived {arrived:.3f}")
```

**Output:**

```text
shortcut False: input kept (cosine) 0.002 | gradient first block 3.55e-02, last block 2.61e-02, ratio 1.360
shortcut True : input kept (cosine) 0.824 | gradient first block 3.05e-02, last block 2.53e-02, ratio 1.209
```

Let's look at the two runs.

**Without shortcuts.** `input kept` is (cosine) `0.002`, while the gradient arriving at the first block is `3.55e-02`. After twelve transformations, the output has almost no directional similarity to the original input, and the backward signal reaching the early layers has become relatively weak. The early layers therefore have a harder time receiving a strong learning signal from the loss.

**With shortcuts.** `input kept` is `0.824`, showing that the output remains strongly aligned with the original input. The gradient at the first block is `3.05e-02`, compared with `2.53e-02` at the last block, so the gradient is also preserved much better across the stack. The identity path carries the original signal forward and provides a direct path for the gradient backward.

Why isnt the cosine similarity exactly `1`? Because the MLPs are still doing the real work. Each block adds its own correction in both directions. The shortcut does not prevent the layers from changing the signal; it simply makes those changes additive instead of forcing each layer to replace everything that came before it.

### The residual stream and the final norm

Follow a single token through all twelve layers. It starts as a vector $\mathbf{x}$. The first layer adds two corrections, the second adds two more, and so on. By the end of the stack the original vector has been joined by 24 separate corrections, one from each sublayer.

This main path through the network is called **residual stream**, which we mentioned in the last chapter ([A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html)). Every sublayer reads from the stream through its norm, and writes back into it by adding its output. The stream itself is never overwritten.

This also explains the gap we left open in the last chapter: the residual stream itself is never normalized. The norms only normalize the input to the sublayers. The shortcut carry the residual stream around those norms, so the stream itself is free to grow.

We can get an intuition for this with a simple example. When adding two unrelated lists of numbers, their variances roughly add. Suppose the stream starts with variance 1, and every layer adds a correction with variance $0.09$ (corresponding to a spread of 0.3). After twelve layers the variance would be about $1 + 12 \cdot 0.09 = 2.08$, so the spread grows from 1 to about $\sqrt{2.08} \approx 1.44$. 

Real corrections are not independent, so this calculation is only a rough illustration, The important point is that every layer adds something to the stream, so the stream can gradually grow.

That is why the encoder ends with one final norm, `post_layernorm`. It brings the 196 vectors in the residual stream back to a steady scale before they leave the encoder and are passed to the language model.

The last chapter mentioned that original Transformer used post norm, which normalizes the stream after every residual addition. We can now see the downside: the direct path through the stack would have to pass through a norm at every layer, so it would no longer be direct. Post norm stacks can still be trained, but they need more care at the beginning of the training, with a learning rate that starts very small and grows step by step, while pre norm stacks can train without it ([Xiong et al., 2020](https://arxiv.org/abs/2002.04745)). The trade-off is that pre-norm architectures need one additional norm at the end of the stack.


![pre norm vs post norm](fig/ch4-pre-vs-post-norm.svg)

### The full vision model

The encoder is a simply a sequence of layers, and the output of one layer becomes the input to the next.

$$\mathbf{x}^{(\ell + 1)} = \text{Layer}_\ell\big(\mathbf{x}^{(\ell)}\big)$$

here, $\ell$ counts the layers from 0 to 11. So the output of the embeddings, $\mathbf{x}^{(0)}$, enters layer 0 and comes out as $\mathbf{x}^{(1)}$, which goes into layer 1, and so on, until $\mathbf{x}^{(12)}$ goes into `post_layernorm`.

Three small classes complete the vision encoder, which we keep in `vision_encoder.py`:

- `VisionEncoder` is the stack of layers.
- `VisionTransformer` puts the embeddings before the stack and the final norm after it.
- `VisionModel` wraps everything together again. This extra wrapper is needed because every vision parameter in the checkpoint starts with `vision_model.`, so our model needs an attribute with the same name.

The layers live in an `nn.ModuleList`, a python list that PyTorch knows how to track. Modules inside it are registered as part of the model, so their parameters are counted, moved to the GPU and loaded from the checkpoint. A regular Python list would not provide this behavior.

```python
class VisionEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [VisionEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(self, inputs_embeds):
        hidden_states = inputs_embeds                             # [B, 196, 768]
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states)          # [B, 196, 768]
        return hidden_states


class VisionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = VisionEmbeddings(config)
        self.encoder = VisionEncoder(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, pixel_values):
        hidden_states = self.embeddings(pixel_values)             # [B, 3, 224, 224] -> [B, 196, 768]
        last_hidden_state = self.encoder(inputs_embeds=hidden_states) # [B, 196, 768]
        return self.post_layernorm(last_hidden_state)             # [B, 196, 768]


class VisionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vision_model = VisionTransformer(config) # this name matches the checkpoint

    def forward(self, pixel_values):
        return self.vision_model(pixel_values=pixel_values)      # [B, 196, 768]
```

Before running it, let's count its parameters by hand. Every number below comes from pieces we have already counted.

| Part              | Calculation                                        | Parameters     |
| ----------------- | -------------------------------------------------- | -------------- |
| Convolution       | $768 \cdot 768$ weights + 768 biases               | 590,592        |
| Position table    | $196 \cdot 768$                                    | 150,528        |
| **Embeddings**    | $590{,}592 + 150{,}528$                            | **741,120**    |
| **One layer**     | two norms $2 \cdot 1536$ + one MLP $4{,}722{,}432$ | **4,725,504**  |
| **Twelve layers** | $12 \cdot 4{,}725{,}504$                           | **56,706,048** |
| **Final norm**    | $2 \cdot 768$                                      | **1536**       |
| **Total**         | $741{,}120 + 56{,}706{,}048 + 1536$                | **57,448,704** |

(Each convolution kernel covers $3 \times 16 \times 16 = 768$ input values, which is why the convolution has $768 \cdot 768$ weights. The stand-in attention still has no parameters.)

```python
torch.manual_seed(0)
model = VisionModel(config)
with torch.no_grad():
    out = model(torch.randn(1, 3, 224, 224))   # [1, 3, 224, 224] -> [1, 196, 768]
print(out.shape)

count = lambda m: sum(p.numel() for p in m.parameters())
vm = model.vision_model
print("embeddings     ", count(vm.embeddings))
print("one layer      ", count(vm.encoder.layers[0]))
print("all 12 layers  ", count(vm.encoder))
print("post_layernorm ", count(vm.post_layernorm))
print("total          ", count(model))
```

**Output:**

```text
torch.Size([1, 196, 768])
embeddings      741120
one layer       4725504
all 12 layers   56706048
post_layernorm  1536
total           57448704
```

A $[1, 3, 224, 224]$ image goes in and produces an output of the shape $[1, 196, 768]$. The five parameter counts matches our calculations from the table.


Most of the parameters are in the **MLPs**. The twelve MLPs hold $12 \cdot 4{,}722{,}432 = 56{,}669{,}184$ out of the total **57,448,704**, about **$99$%** . That will change in the next chapter, when attention adds four more linear layers to every layer, but even then the MLP will hold about two thirds of each layer's parameters.

### Our names against the real checkpoint

Before we load the trained weights, there is one more thing to verify: every parameter name in our model must match the corresponding name in the checkpoint.

The first list comes from the checkpoint. A `.safetensors` file contains a small header describing every tensor inside it, including its name and shape. `get_safetensors_metadata` can read that header without loading the tensor values.

The checkpoint contains more than just the vision encoder. It also contains a text encoder and a small pooling head at the end of the vision model that we do not need here. We therefore keep only the vision tensors and exclude the pooling head.

The second list comes from our own model, built using the real model's dimensions. We build it on PyTorch's `meta` device, which creates tensors with shapes but without allocating storage for their values. This lets us inspect a model of this size without using significant memory.

```python
import re
from huggingface_hub import get_safetensors_metadata

meta = get_safetensors_metadata("google/siglip-so400m-patch14-224")
ckpt = {name: tuple(info.shape)
        for f in meta.files_metadata.values()
        for name, info in f.tensors.items()
        if name.startswith("vision_model.") and not name.startswith("vision_model.head.")}   # the pooling head we skip

real = VisionConfig(hidden_size=1152, intermediate_size=4304, num_hidden_layers=27,
                    num_attention_heads=16, patch_size=14)
with torch.device("meta"):
    ours = VisionModel(real)                                      # shapes only, no numbers
mine = {k: tuple(v.shape) for k, v in ours.state_dict().items()}

missing = sorted(set(ckpt) - set(mine))
extra = sorted(set(mine) - set(ckpt))
wrong = sorted(k for k in set(ckpt) & set(mine) if ckpt[k] != mine[k])
pattern = lambda keys: sorted({re.sub(r"layers\.\d+\.", "layers.N.", k) for k in keys})   # one line per kind of tensor

print("checkpoint vision tensors:", len(ckpt), "| ours:", len(mine))
print("missing in ours:", len(missing), pattern(missing))
print("extra in ours:  ", extra)
print("wrong shape:    ", wrong)
```

**Output:**

```text
checkpoint vision tensors: 437 | ours: 221
missing in ours: 216 ['vision_model.encoder.layers.N.self_attn.k_proj.bias', 'vision_model.encoder.layers.N.self_attn.k_proj.weight', 'vision_model.encoder.layers.N.self_attn.out_proj.bias', 'vision_model.encoder.layers.N.self_attn.out_proj.weight', 'vision_model.encoder.layers.N.self_attn.q_proj.bias', 'vision_model.encoder.layers.N.self_attn.q_proj.weight', 'vision_model.encoder.layers.N.self_attn.v_proj.bias', 'vision_model.encoder.layers.N.self_attn.v_proj.weight']
extra in ours: []
wrong shape: []
```

Each norm and each linear layer has two tensors: a weight and a bias. A checkpoint layer has 2 norms, 4 attention layers and 2 MLP layers, so $2 \cdot 8 = 16$ tensors, and 27 layers have $27 \cdot 16 = 432$. Add 3 for the embeddings and 2 for `post_layernorm`, and the checkpoint has 437 vision tensors.

Our current layers do not have attention yet, so each one has 8 tensors instead of 16. That gives $27 \cdot 8 = 216$, plus the same 5 tensors outside the layers, for a total of 221.

The 216 missing tensors are the attention weights: a `weight` and a `bias` for each of `self_attn.k_proj`, `q_proj`, `v_proj` and `out_proj`, in all 27 layer. That is exactly what the next chapter adds. The `extra` and `wrong shape` lists are both empty, which meansevery parameter name currently implemented in our model exists in the checkpoint with the expected shape. (The `position_ids` buffer does not appear in either list because it was created with `persistent=False`, so it is not saved as a model parameter, just as we discussed in chapter 2.

Notice that we never typed the real model's dimensions directly into the classes. The values such as `fc1` with shape $[4304,1152]$, the position table with shape $[256,1152]$, and the convolution with shape $[1152,3,14,14]$ all come from the config. The same classes build both our small model and the real model; only the config changes.

### A real photo through a trained encoder

Our implementation still uses random weights, so it cannot tell us much about what a trained encoder does to a real image. The ViT we loaded in the previous two chapters can.

Its layers follow the same structure as ours: a norm, attention, a shortcut, another norm, the MLP, and another shortcut. Some of the parameter names are different, though. Hugging Face calls the two MLP linear layers `intermediate.dense` and `output.dense`, and it calls the second norm `layernorm_after`.

The block below performs two checks.

First, it copies the trained MLP from the ViT's first layer into our `VisionMLP` and compares their outputs. This ViT was trained with the exact GELU, while our class uses the tanh approximation, so we should expect a very small difference rather than exactly zero.

Second, it runs the dog image through all 12 trained layers and records the residual stream after each one. As before, we drop the CLS token. For every layer, we measure two things: the spread of the residual stream and the cosine similarity between the stream before and after the layer. The latter tells us how much of the layer's input remains in its output.

```python
layer0 = next(m for n, m in vit.named_modules() if re.search(r"(^|\.)layers?\.0$", n))   # layer 0, any version
inner = list(layer0.named_modules())

fc1_name, fc1_real = next((n, m) for n, m in inner if isinstance(m, nn.Linear)
                          and (m.in_features, m.out_features) == (768, 3072))           # expand
fc2_name, fc2_real = next((n, m) for n, m in inner if isinstance(m, nn.Linear)
                          and (m.in_features, m.out_features) == (3072, 768))           # compress
lns = [(n, m) for n, m in inner if isinstance(m, nn.LayerNorm)]
ln2_name, ln2_real = next(((n, m) for n, m in lns if "after" in n), lns[-1])            # the norm before the MLP

print("fc1:", fc1_name, "| fc2:", fc2_name, "| norm:", ln2_name, "| all norms:", [n for n, _ in lns])

mlp_real = VisionMLP(VisionConfig()).eval() # your class, trained weights copied in
with torch.no_grad():
    mlp_real.fc1.weight.copy_(fc1_real.weight); mlp_real.fc1.bias.copy_(fc1_real.bias)
    mlp_real.fc2.weight.copy_(fc2_real.weight); mlp_real.fc2.bias.copy_(fc2_real.bias)

    out = vit(pix, output_hidden_states=True)
    streams = [h[0, 1:] for h in out.hidden_states] # 13 x [196, 768], before layer 0, then after each layer, CLS dropped

    x_in = ln2_real(streams[0])              # [196, 768], a real normalized input
    ours = mlp_real(x_in)                    # [196, 768], tanh GELU
    ref = fc2_real(F.gelu(fc1_real(x_in)))   # [196, 768], exact GELU, as this ViT was trained
print(f"our MLP vs the trained ViT's MLP, largest difference: {(ours - ref).abs().max():.6f}")
print(f"typical size of the MLP output:                     {ref.abs().mean():.4f}")

spread = [s.std(dim=-1, unbiased=False).mean().item() for s in streams]          # 13 values
kept = [F.cosine_similarity(streams[i], streams[i + 1], dim=-1).mean().item()    # 12 values
        for i in range(len(streams) - 1)]
final = out.last_hidden_state[0, 1:].std(dim=-1, unbiased=False).mean().item()   # after the final norm

print(f"{'':10s} spread of stream   cosine with previous")
print(f"{'input':10s} {spread[0]:16.3f}")
for i in range(len(kept)):
    print(f"{'layer ' + str(i):10s} {spread[i + 1]:16.3f} {kept[i]:20.3f}")
print(f"{'final norm':10s} {final:16.3f}")

fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
ax[0].plot(range(13), spread, marker="o"); ax[0].axhline(final, ls="--", c="grey", label="after final norm")
ax[0].set_xlabel("layers done"); ax[0].set_title("spread of the main stream"); ax[0].legend()
ax[1].plot(range(1, 13), kept, marker="o"); ax[1].set_ylim(0, 1)
ax[1].set_xlabel("layer"); ax[1].set_title("how much of its input a layer keeps (cosine)")
plt.tight_layout(); plt.show()
```

**Output:**

```text
fc1=mlp.fc1 | fc2=mlp.fc2 | norm=layernorm_after
all norms=['layernorm_before', 'layernorm_after']

MLP difference: 0.010559
Typical MLP output size: 0.8526

             spread    cosine_prev
input        0.768          —
layer 0      1.114       0.638
layer 1      1.242       0.787
layer 2      1.464       0.799
layer 3      1.668       0.863
layer 4      2.187       0.855
layer 5      3.905       0.875
layer 6      5.298       0.900
layer 7      5.701       0.899
layer 8      6.246       0.849
layer 9      6.973       0.873
layer 10     8.431       0.818
layer 11    10.653       0.761
final norm   0.902
```


**MLP.** The largest difference between VisionMLP and the trained ViT's MLP is **0.010559**, while a typical output value has a magnitude of about **0.8526**. The small difference comes from using the tanh approximation instead of the exact GELU. Apart from that approximation, our MLP is reproducing the trained module correctly.

**The spread column.** The residual stream starts with a spread of **0.768** and reaches **10.653** after the last layer. **The spread grows steadily layer by layer, increasing by roughly 14× overall, although the amount of growth varies from layer to layer.** This is what we would expect from a residual stream that keeps accumulating corrections. Nothing inside the stack forces the stream itself back to a fixed scale. The final norm changes that: the last line, **0.902**, shows the stream after post_layernorm has brought it back to a steady scale.

**The cosine column.** The cosine values range from **0.638** to **0.900**. Every layer therefore keeps a substantial part of the vector it receives. The trained ViT is not throwing away the representation at every layer and starting over. It keeps the existing stream and adds new information to it.

Unlike our stand-in, of course, these layers contain real attention. Some of those added corrections therefore come from other patches.

### Where the new pieces sit

Here is the encoder again, with everything we have built so far, shortcuts included.

```text
  image              [B, 3, 224, 224]
  embeddings         [B, 196, 768]
  layer 0            h = x + self_attn(layer_norm1(x))
                     y = h + mlp(layer_norm2(h))          [B, 196, 768]
  layer 1            same                                 [B, 196, 768]
  ...
  layer 11           same                                 [B, 196, 768]
  post_layernorm     [B, 196, 768]
```

Everything shown here is implemented except: `self_attn` is still the stand-in that returns zeros.

![ch4-so-far](fig/ch4-vision-model-tree.svg)

Our encoder now has its complete skeleton: twelve layers, each with two norms and two shortcuts, an MLP that processes each patch independently, and a final norm on the residual stream.

The editors are in place, and the page can now travel safely from the first editor to the last. But there is still something missing: the editors cannot share information yet. Look at what each patch can see. The norm reads one token at a time. The MLP also reads one token. The ear patch goes through all twelve layers and still knows nothing about the snout, because the only part that lets patches talk currently returns zeros. That is what we will fix in the next chapter. We will replace the stand-in with **real attention**, the part of the layer where every patch finally looks at every other patch.

---
