# Multimodal-Question-Answering

This is the codebase for CMU MCDS 11-632 and 11-635 Data Science Capstone, Multimodal Question Answering team.

This project is conducted by Jialu Sui, Naveen Suresh, Ninaad R. Rao, Onkar Thorat, Pragnya Sridhar, Pratik Mandlecha, Yujia Wang, and advised by Dr. Eric Nyberg.

### Overview
* This research project aims to enhance Visual Question Answering (VQA) systems by addressing challenges in three key dimensions.
* Firstly, it focuses on improving the consistency and effectiveness of visual question-answering models across real-world and synthetically-generated images, narrowing the performance gap between these domains. The relevant code can be found in `thrust1/`.
* Secondly, the integration of scene graphs is explored to enhance question-answering accuracy, particularly for questions related to object relationships and counting, fostering a more nuanced understanding of spatial relationships within images. The relevant code can be found in `thrust2/`.
* Thirdly, the project aims to augment the system's capacity for answering questions requiring external knowledge by leveraging latent information from Language Models (LLMs) and employing efficient few-shot inference techniques. The relevant code can be found in `thrust3/`.
* Finally, these improvements are unified through a Mixture of Experts (MoE) framework, allowing for the integration of diverse expertise from individual models, creating a cohesive and robust visual question-answering system capable of accurately addressing a wide array of questions across synthetic and real-world scenarios, promising advancements in the field of Multimodal Question Answering. The relevant code can be found in `router/`.

