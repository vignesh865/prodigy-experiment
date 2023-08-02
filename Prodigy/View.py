import streamlit as st
import random
import time

from sap_gpt.SapGPTClass import SapGpt

sapGPT = SapGpt.get_instance(init_prodigy=True)

st.title("Knowledge companion")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        # assistant_response = random.choice(
        #     [
        #         "Hello there! How can I assist you today?",
        #         "Hi, human! Is there anything I can help you with?",
        #         "Do you need help?",
        #     ]
        # )

        assistant_response = sapGPT.answer_with_chain(prompt)["result"]

        # Simulate stream of response with milliseconds delay
        for sentence in assistant_response.split("\n"):
            for chunk in sentence.split():
                full_response += chunk + " "
                time.sleep(0.05)
                # Add a blinking cursor to simulate typing
                message_placeholder.markdown(full_response + "▌")
            full_response += "\n"

        message_placeholder.markdown(full_response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})




# What are the advantages of connecting different applications to the directory services ?
# How to work with directory services in cua?
# How to define RFC connections in the transaction SM59?



