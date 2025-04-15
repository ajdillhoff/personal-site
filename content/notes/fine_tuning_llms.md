+++
title = "Fine-Tuning LLMs"
authors = ["Alex Dillhoff"]
date = 2025-04-15T10:37:00-05:00
draft = false
sections = "Machine Learning"
lastmod = 2025-04-15
+++

<div class="ox-hugo-toc toc">

<div class="heading">Table of Contents</div>

- [Supervised Fine-Tuning](#supervised-fine-tuning)
- [Reinforcement Learning with Human Feedback](#reinforcement-learning-with-human-feedback)
- [Direct Preference Optimization](#direct-preference-optimization)
- [Using Tools](#using-tools)
- [Chain of Thought (CoT) and Reasoning](#chain-of-thought--cot--and-reasoning)

</div>
<!--endtoc-->

After pre-training an LLM, which builds a general model of language understanding, fine-tuning is necessary to adapt the model to behave like a helpful assistant.


## Supervised Fine-Tuning {#supervised-fine-tuning}

The first form of post-training is **Supervised Fine-Tuning (SFT)**. This requires annotated examples of prompts and responses that demonstrate _how_ the model should behave when responding to user input.

{{< figure src="/ox-hugo/2025-04-15_10-45-00_screenshot.png" caption="<span class=\"figure-number\">Figure 1: </span>SFT pipeline (<a href=\"#citeproc_bib_item_6\">Ouyang et al. 2022</a>)." >}}

These datasets are much more expensive to create since they require more thought in the quality of prompts and responses. Pre-training may be cheap with regards to annotation, but it is expensive in the cost of curation and compute. SFT is cheap in terms of compute but requires expensive annotations.

It is impossible to create a dataset with every conceivable question and answer. Instead, the model only needs to mimic the _style_ of responses seen in the dataset. An open source dataset for SFT can be viewed [here](https://huggingface.co/datasets/OpenAssistant/oasst2).


## Reinforcement Learning with Human Feedback {#reinforcement-learning-with-human-feedback}

Reinforcement learning is used to improve the _quality_ of responses from an LLM. Instead of requiring a human annotator to provide ideal responses to all conceivable prompts, they only need to train a model to mimic their preferences. This is typically done after the SFT stage.

{{< figure src="/ox-hugo/2025-04-15_10-53-16_screenshot.png" caption="<span class=\"figure-number\">Figure 2: </span>RLHF pipeline (<a href=\"#citeproc_bib_item_6\">Ouyang et al. 2022</a>)" >}}

This type of training was also applied to text summarization, where a human annotator would rate the best summarization provided from a model (<a href="#citeproc_bib_item_9">Stiennon et al. 2022</a>).

In recent models, it is common to have a previous checkpoint generate prompts and responses that are then evaluated by a human annotator (<a href="#citeproc_bib_item_4">DeepSeek-AI et al. 2025</a>). Llama 3 uses a mixture of human annotations with synthetic data (<a href="#citeproc_bib_item_5">Dubey et al. 2024</a>).


## Direct Preference Optimization {#direct-preference-optimization}

RLHF has proven effective for fine-tuning models to provide high quality and accurate responses. However, the additional training required can be replaced by Direct Preference Optimization (DPO) (<a href="#citeproc_bib_item_7">Rafailov et al. 2024</a>). In this work, the preference data is input directly into the model and trains it using a simple classification objective.

{{< figure src="/ox-hugo/2025-04-15_11-10-25_screenshot.png" caption="<span class=\"figure-number\">Figure 3: </span>RLHF vs. DPO (<a href=\"#citeproc_bib_item_7\">Rafailov et al. 2024</a>)" >}}


## Using Tools {#using-tools}

Adding tool use expands the capabilities and accuracy of LLMs on a wide range of tasks. For example, being able to verify information reduces the likelihood of hallucinations and general misinformation. Models can adapt to tool use via self-supervised learning, where a few examples of the tools and how to use them are all that is needed (<a href="#citeproc_bib_item_8">Schick et al. 2023</a>).

{{< figure src="/ox-hugo/2025-04-15_11-16-30_screenshot.png" caption="<span class=\"figure-number\">Figure 4: </span>Examples of tool use predictions (<a href=\"#citeproc_bib_item_8\">Schick et al. 2023</a>)." >}}

A small dataset of prompts is created which includes example behavior and special syntax for calling and executing tools. This is also known as few-shot learning (<a href="#citeproc_bib_item_1">Brown et al. 2020</a>), a capability that LLMs already have from pre-training. Unique prompts are required for each tool.

{{< figure src="/ox-hugo/2025-04-15_11-19-37_screenshot.png" caption="<span class=\"figure-number\">Figure 5: </span>Example few-shot prompt for tool use (<a href=\"#citeproc_bib_item_8\">Schick et al. 2023</a>)." >}}

Once the model has examples of which tools are available and how to use them, it can sample positions from input text and generate API calls to the tools. The result is then evaluated using a loss function. If the loss is not reduced, the tool call is filtered out.

{{< figure src="/ox-hugo/2025-04-15_11-25-14_screenshot.png" caption="<span class=\"figure-number\">Figure 6: </span>Self-supervised pipeline of Toolformer (<a href=\"#citeproc_bib_item_8\">Schick et al. 2023</a>)." >}}


## Chain of Thought (CoT) and Reasoning {#chain-of-thought--cot--and-reasoning}

It has been shown that prompting a model to explain its calculations lead to higher accuracy on a variety of tasks (<a href="#citeproc_bib_item_10">Wei et al. 2023</a>), (<a href="#citeproc_bib_item_2">Chung et al. 2022</a>). This type of instruction is facilitated through in-context learning on example prompts.

{{< figure src="/ox-hugo/2025-04-15_11-44-49_screenshot.png" caption="<span class=\"figure-number\">Figure 7: </span>Example of CoT prompting (<a href=\"#citeproc_bib_item_10\">Wei et al. 2023</a>)." >}}

CoT prompts elicit reasoning behaviors in LLMs. To train a model to exhibit more reasoning on complex tasks, a reward model was employed which provides a higher reward when the model produces longer responses. As opposed to traditional RLHF, where the output is subjective and is only rated against human preferences, verifiable data is used for training. This allows the model to immediately determine if it was correct or not (<a href="#citeproc_bib_item_3">DeepSeek-AI 2025</a>).

{{< figure src="/ox-hugo/2025-04-15_11-52-25_screenshot.png" caption="<span class=\"figure-number\">Figure 8: </span>The reward model encourages longer responses when solving complex tasks (<a href=\"#citeproc_bib_item_3\">DeepSeek-AI 2025</a>)." >}}

## References

<style>.csl-entry{text-indent: -1.5em; margin-left: 1.5em;}</style><div class="csl-bib-body">
  <div class="csl-entry"><a id="citeproc_bib_item_1"></a>Brown, Tom B., Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, et al. 2020. “Language Models Are Few-Shot Learners.” arXiv. <a href="https://doi.org/10.48550/arXiv.2005.14165">https://doi.org/10.48550/arXiv.2005.14165</a>.</div>
  <div class="csl-entry"><a id="citeproc_bib_item_2"></a>Chung, Hyung Won, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan Li, et al. 2022. “Scaling Instruction-Finetuned Language Models.” arXiv. <a href="https://doi.org/10.48550/arXiv.2210.11416">https://doi.org/10.48550/arXiv.2210.11416</a>.</div>
  <div class="csl-entry"><a id="citeproc_bib_item_3"></a>DeepSeek-AI. 2025. “DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.”</div>
  <div class="csl-entry"><a id="citeproc_bib_item_4"></a>DeepSeek-AI, Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, et al. 2025. “DeepSeek-V3 Technical Report.” arXiv. <a href="https://doi.org/10.48550/arXiv.2412.19437">https://doi.org/10.48550/arXiv.2412.19437</a>.</div>
  <div class="csl-entry"><a id="citeproc_bib_item_5"></a>Dubey, Abhimanyu, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, et al. 2024. “The Llama 3 Herd of Models.” arXiv.</div>
  <div class="csl-entry"><a id="citeproc_bib_item_6"></a>Ouyang, Long, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, et al. 2022. “Training Language Models to Follow Instructions with Human Feedback.” arXiv. <a href="https://doi.org/10.48550/arXiv.2203.02155">https://doi.org/10.48550/arXiv.2203.02155</a>.</div>
  <div class="csl-entry"><a id="citeproc_bib_item_7"></a>Rafailov, Rafael, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, and Chelsea Finn. 2024. “Direct Preference Optimization: Your Language Model Is Secretly a Reward Model.” arXiv. <a href="https://doi.org/10.48550/arXiv.2305.18290">https://doi.org/10.48550/arXiv.2305.18290</a>.</div>
  <div class="csl-entry"><a id="citeproc_bib_item_8"></a>Schick, Timo, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. 2023. “Toolformer: Language Models Can Teach Themselves to Use Tools.” arXiv. <a href="https://doi.org/10.48550/arXiv.2302.04761">https://doi.org/10.48550/arXiv.2302.04761</a>.</div>
  <div class="csl-entry"><a id="citeproc_bib_item_9"></a>Stiennon, Nisan, Long Ouyang, Jeff Wu, Daniel M. Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, and Paul Christiano. 2022. “Learning to Summarize from Human Feedback.” arXiv. <a href="https://doi.org/10.48550/arXiv.2009.01325">https://doi.org/10.48550/arXiv.2009.01325</a>.</div>
  <div class="csl-entry"><a id="citeproc_bib_item_10"></a>Wei, Jason, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, and Denny Zhou. 2023. “Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.” arXiv. <a href="https://doi.org/10.48550/arXiv.2201.11903">https://doi.org/10.48550/arXiv.2201.11903</a>.</div>
</div>
