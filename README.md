  # sSanityLayer
A multi head attention layer built for SLMs with semantic anchoring for emotional variance.

# What is sSanityLayer?

For humans, sanity influences emotion. For an AI, emotions are vectors. ```sSanityLayer``` introduces a separate attention layer in parallel to the self attention layer found in open source model architectures.

# Features

```sSanityLayer``` can do two of the following:

1) Using its modified hybrid ```RNN``` structure over the pre nanoGPT architecture, ```sSanityLayer``` can train very small language models ```(vSLMs)```.

   It replaces the standard Self-Attention "Value" projection with a **GRU (Gated Recurrent Unit)** to mix sequential history into the attention mechanism.

>   This architecture can be used to build models extremely small in size with alterable states of ```Sane``` and ```Insane```. A model could respond negatively to prompts, just like the human mind does.

**A demonstration of the same is a trained model from scratch named ```'Potato'```.** 

A model so tiny, with only **16.95K** parameters at **77KB**

In the Potato Architecture, the Linear layer for values is replaced by a Gated Recurrent Unit (GRU).

**Standard Attention**:

$Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V_{linear}$

**Altered Attention**: 

$Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V_{GRU}$

For a given ```RNN```, $$h_t = f(h_{t-1}, x_t)$$

When ```Sanity``` drops below a threshold, 

$$\lim_{S \to 0} h_{t-1} \equiv \mathcal{H}_{\text{max}}$$

That is, the vector noise is maximum due to loss of vector positions. This allows for a model, particularly a ```vSLM```, to react with speech till the sanity is alive.

This is achieved by ```logits[tid] += boost``` where ```boost``` is just a variable with strength.

Below is the representiation of ```Potato``` trained on sSanityLayer.

  ![alt text](https://github.com/kavyamali/sSanityLayer/blob/main/Graph.png)



2) **Vector Injection**

A ```SLM``` lacks the ability to reproduce emotional states, when it relies purely on training data and **SelfAttention**.

Unlike the sSanityLayer used for the Potato architecture, the Vector injection method can be used on any standard ```transformers``` package.

* **Mechanism:** Intercepts the output logits (prediction vectors) before generation.
* **Dynamic Anchoring:** Automatically maps semantic concepts (e.g., "fear") to the specific token IDs of the target model (whether GPT-2, LLaMA, or others) at runtime.
* **Effect:** Allows for controlled "hallucination" by mathematically boosting specific emotional vectors during inference.

* **Example:**

**GPT2 Small(Without ```sSanityLayer```):**

```
Prompt: "a girl died in the woods once, her family wept"
Output: "...over the night. Features of Venus found this... Destiny 2 images... omnipresent spirits of dragons... Reign of Chaos... New Blood Paradox."
```

```
Prompt: "the girl died in the woods, her family wept"
Output: "She had disguised herself as Khalid and spent the next two years living as a Goonafilitator... exterminating 'superstitious jihadi natural warriors'."
```

**GPT2 Small(With ```sSanityLayer```):**

```
Prompt: "a girl died in the woods once, her family wept"
Output: "(then) after leaving the house. No one ever forgave her for lack of a body."
```

```
Prompt: "the girl died in the woods, her family wept"
Output: "...as they tried to locate her spot," her grandmother Kale Soukaki told a Thai newspaper.
```

> The layer introduces a vector bias for tokens actively, and on lower sanity levels, uses the data from the ```anchors``` to further bias output decisively.

Using Byte-Level BPE, it treats " word" (with a space) and "word" (without) as different tokens:

```
t_ids_2 = tokenizer.encode(" " + w, add_special_tokens=False)
```

# Testing

Clone the repository, and you can test the potato model demonstration with only ```pytorch``` installed.

```
pip install -r requirements.txt
python sSanityLayerdemo.py
```

> You need to run GPT2Demo/gpt2.py atleast once to install the dependencies trying it.
