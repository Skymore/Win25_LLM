## Multi-Agent Chatbot Project Documentation

## System Architecture

<img src="./system.svg" alt="System Architecture" style="max-height:600px;">

## Handwritten System Architecture

<img src="./image.png" alt="System Architecture" style="max-height:600px;">

---

## Project Challenges and Insights

While working on this multi-agent chatbot project, I faced several challenges and gained valuable insights along the way. One of the biggest challenges was handling prompt injection and detecting obnoxious content. Since I was limited to only two LLM API calls, designing a system that could accurately identify malicious or irrelevant queries while maintaining efficiency was quite difficult. To overcome this, I had to carefully craft input prompts and combine multiple tasks—such as detecting obnoxious content, verifying query relevance, and preventing prompt injection—into a single API call. This required a deep understanding of prompt engineering and extensive testing to ensure the system's reliability.

Through this assignment, I learned how to implement complex functionalities without relying on specific APIs and with limited resources. This experience deepened my understanding of prompt engineering techniques and how to optimize the efficiency of LLM calls. Additionally, I gained an appreciation for the importance of modular design. By separating different functionalities into independent agents, the system became easier to maintain and extend while also improving overall flexibility. These experiences have provided me with a solid foundation for developing more advanced AI systems in the future.

---

## Implementation Details

### Agent Components

1. **Head Agent**
   - System coordinator
   - Manages query flow
   - Coordinates between sub-agents

2. **Filtering Agents**
   - Security check
   - ML relevance verification
   - Efficient dual-purpose filtering

3. **Query Agent**
   - Vector similarity search
   - Document retrieval
   - Context management

4. **Answering Agent**
   - Response generation
   - Context integration
   - Dual response modes

### Technical Specifications

- Maximum 3 LLM API calls per query
- Vector-based document retrieval
- Streamlit web interface
- Modular agent architecture

---

## Screenshots - Streamlit Interface

![Streamlit Interface](./streamlit1.png)

![Streamlit Interface2](./streamlit2.png)

![Streamlit Interface3](./streamlit3.png)


---

## Future Improvements

1. Enhanced document retrieval accuracy
2. Expanded response customization
3. Improved conversation context handling
4. Advanced security measures
