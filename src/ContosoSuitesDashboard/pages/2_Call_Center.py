import streamlit as st
import streamlit_extras.stateful_button as stx
from streamlit_js_eval import streamlit_js_eval
import requests
import pandas as pd
import json
import openai
import inspect
import time
from scipy.io import wavfile
import azure.cognitiveservices.speech as speechsdk
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import (
        TextAnalyticsClient,
        ExtractiveSummaryAction,
        AbstractiveSummaryAction
    )

st.set_page_config(layout="wide")

with open('config.json') as f:
    config = json.load(f)

aoai_endpoint = config['AOAIEndpoint']
aoai_api_key = config['AOAIKey']
deployment_name = config['AOAIDeploymentName']
speech_key = config['SpeechKey']
speech_region = config['SpeechRegion']
language_endpoint = config['LanguageEndpoint']
language_key = config['LanguageKey']

### Exercise 05: Provide live audio transcription
def create_transcription_request(audio_file, speech_key, speech_region, speech_recognition_language="en-US"):
    # Create an instance of a speech config with specified subscription key and service region.

    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
    speech_config.speech_recognition_language=speech_recognition_language

    # Prepare audio settings for the wave stream
    channels = 1
    bits_per_sample = 16
    samples_per_second = 16000

    # Create audio configuration using the push stream
    wave_format = speechsdk.audio.AudioStreamFormat(samples_per_second, bits_per_sample, channels)
    stream = speechsdk.audio.PushAudioInputStream(stream_format=wave_format)
    audio_config = speechsdk.audio.AudioConfig(stream=stream)

    transcriber = speechsdk.transcription.ConversationTranscriber(speech_config, audio_config)
    all_results = []

    def handle_final_result(evt):
        all_results.append(evt.result.text)

    done = False

    def stop_cb(evt):
        print('CLOSING on {}'.format(evt))
        nonlocal done
        done= True

    # Subscribe to the events fired by the conversation transcriber
    transcriber.transcribed.connect(handle_final_result)
    transcriber.session_started.connect(lambda evt: print('SESSION STARTED: {}'.format(evt)))
    transcriber.session_stopped.connect(lambda evt: print('SESSION STOPPED {}'.format(evt)))
    transcriber.canceled.connect(lambda evt: print('CANCELED {}'.format(evt)))
    # stop continuous transcription on either session stopped or canceled events
    transcriber.session_stopped.connect(stop_cb)
    transcriber.canceled.connect(stop_cb)

    transcriber.start_transcribing_async()

    # Read the whole wave files at once and stream it to sdk
    _, wav_data = wavfile.read(audio_file)
    stream.write(wav_data.tobytes())
    stream.close()
    while not done:
        time.sleep(.5)

    transcriber.stop_transcribing_async()
    return all_results


def create_live_transcription_request(speech_key, speech_region, speech_recognition_language="en-US"):

    # Creates speech configuration with subscription information
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
    speech_config.speech_recognition_language=speech_recognition_language
    transcriber = speechsdk.transcription.ConversationTranscriber(speech_config)

    done = False

    def handle_final_result(evt):
        all_results.append(evt.result.text)
        print(evt.result.text)

    all_results = []

    def stop_cb(evt: speechsdk.SessionEventArgs):
        """callback that signals to stop continuous transcription upon receiving an event `evt`"""
        print('CLOSING {}'.format(evt))
        nonlocal done
        done = True

    # Subscribe to the events fired by the conversation transcriber
    transcriber.transcribed.connect(handle_final_result)
    transcriber.session_started.connect(lambda evt: print('SESSION STARTED: {}'.format(evt)))
    transcriber.session_stopped.connect(lambda evt: print('SESSION STOPPED {}'.format(evt)))
    transcriber.canceled.connect(lambda evt: print('CANCELLED {}'.format(evt)))
    # stop continuous transcription on either session stopped or canceled events
    transcriber.session_stopped.connect(stop_cb)
    transcriber.canceled.connect(stop_cb)

    transcriber.start_transcribing_async()

    # Streamlit refreshes the page on each interaction,
    # so a clean start and stop isn't really possible with button presses.
    # Instead, we're constantly updating transcription results, so that way,
    # when the user clicks the button to stop, we can just stop updating the results.
    # This might not capture the final message, however, if the user stops before
    # we receive the message--we won't be able to call the stop event.
    while not done:
        st.session_state.transcription_results = all_results
        time.sleep(1)

    return


def make_azure_openai_chat_request(system, call_contents):
    # Create an Azure OpenAI client.
    client = openai.AzureOpenAI(
        base_url=f"{aoai_endpoint}/openai/deployments/{deployment_name}/",
        api_key=aoai_api_key,
        api_version="2023-12-01-preview"
    )

    # Create and return a new chat completion request
    return client.chat.completions.create(
        model=deployment_name,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": call_contents}
        ],
    )


def is_call_in_compliance(call_contents, include_recording_message, is_relevant_to_topic):
    joined_call_contents = ' '.join(call_contents)
    if include_recording_message:
        include_recording_message_text = "2. Was the caller aware that the call was being recorded?"
    else:
        include_recording_message_text = ""

    if is_relevant_to_topic:
        is_relevant_to_topic_text = "3. Was the call relevant to the hotel and resort industry?"
    else:
        is_relevant_to_topic_text = ""

    system = f"""
        You are an automated analysis system for Contoso Suites. Contoso Suites is a luxury hotel and resort chain with locations
        in a variety of Caribbean nations and territories.
            
        You are analyzing a call for relevance and compliance.

        You will only answer the following questions based on the call contents:
        1. Was there vulgarity on the call?
        {include_recording_message_text}
        {is_relevant_to_topic_text}
    """

    response = make_azure_openai_chat_request(system, joined_call_contents)
    return response.choices[0].message.content



### Exercise 06: Generate call summaries
def generate_extractive_summary(call_contents):
    # TODO:
    # 1. The call_contents parameter is formatted as a list of strings. Join them together with spaces to pass in as a single document.
    
    # 2. Create a TextAnalyticsClient, connecting it to your Language Service endpoint.
    # client = ...

    # 3. Complete the call the begin_analyze_actions method on your client, passing in the joined call_contents as an array
    #    and an ExtractiveSummaryAction with a max_sentence_count of 2.
    # poller = client.begin_analyze_actions(
    #   ...
    #)

    #   4. Extract the summary result sentences and merge them into a single summary string.
    # for result in poller.result():
    #   ...

    #   5. Return the summary as a JSON object in the shape '{"call-summary": extractive_summary}'.
    raise NotImplementedError

def generate_abstractive_summary(call_contents):
    # TODO:
    # 1. The call_contents parameter is formatted as a list of strings. Join them together with spaces to pass in as a single document.

    # 2. Create a TextAnalyticsClient, connecting it to your Language Service endpoint.
    # client = ...
    
    # 3. Call the begin_analyze_actions method on your client, passing in the joined call_contents as an array
    #    and an AbstractiveSummaryAction with a sentence_count of 2.
    # poller = client.begin_analyze_actions(
    #   ...
    #)
    
    # 4. Extract the summary result summaries and merge them into a single summary string.
    # for result in poller.result():
    #   ...
    
    # 5. Return the summary as a JSON object in the shape '{"call-summary": abstractive_summary}'.
    raise NotImplementedError

def generate_query_based_summary(call_contents):
    # TODO:
    # 1. The call_contents parameter is formatted as a list of strings. Join them together with spaces to pass in as a single document.

    # 2. Write a system prompt that instructs the large language model to:
    #    - Generate a short (5 word) summary from the call transcript.
    #    - Create a two-sentence summary of the call transcript.
    #    - Output the response in JSON format, with the short summary labeled 'call-title' and the longer summary labeled 'call-summary.'
    system = f"""
    """

    # 3. Call make_azure_openai_chat_request().
    # response = ...

    # 4. Return the message content for the response's first choice.

    # 5. Return the summary.
    raise NotImplementedError

def create_sentiment_analysis_and_opinion_mining_request(call_contents):
    # TODO:
    # 1. The call_contents parameter is formatted as a list of strings. Join them together with spaces to pass in as a single document.

    # 2. Create a Text Analytics Client
    #client = ...

    # 3. Analyze sentiment of call transcript, enabling opinion mining.
    #result = client...

    # 4. Retrieve all document results that are not an error.
    doc_result = None #...

    # The output format is a JSON document with the shape:
    # {
    #     "sentiment": document_sentiment,
    #     "sentiment-scores": {
    #         "positive": document_positive_score_as_two_decimal_float,
    #         "neutral": document_neutral_score_as_two_decimal_float,
    #         "negative": document_negative_score_as_two_decimal_float
    #     },
    #     "sentences": [
    #         {
    #             "text": sentence_text,
    #             "sentiment": document_sentiment,
    #             "sentiment-scores": {
    #                 "positive": document_positive_score_as_two_decimal_float,
    #                 "neutral": document_neutral_score_as_two_decimal_float,
    #                 "negative": document_negative_score_as_two_decimal_float
    #             },
    #             "mined_opinions": [
    #                 {
    #                     "target-sentiment": opinion_sentiment,
    #                     "target-text": opinion_target,
    #                     "target-scores": {
    #                         "positive": document_positive_score_as_two_decimal_float,
    #                         "neutral": document_neutral_score_as_two_decimal_float,
    #                         "negative": document_negative_score_as_two_decimal_float
    #                     },
    #                     "assessments": [
    #                       {
    #                         "assessment-sentiment": assessment_sentiment,
    #                         "assessment-text": assessment_text,
    #                         "assessment-scores": {
    #                             "positive": document_positive_score_as_two_decimal_float,
    #                             "negative": document_negative_score_as_two_decimal_float
    #                         }
    #                       }
    #                     ]
    #                 }
    #             ]
    #         }
    #     ]
    # }
    sentiment = {}

    # 5. Assign the correct values to the JSON object.
    for document in doc_result:
        #sentiment["sentiment"] = document...
        sentiment["sentiment-scores"] = {
            #"positive": document...,
            #"neutral": document...,
            #"negative": document...
        }
        
        sentences = []
        for s in document.sentences:
            sentence = {}
            #sentence["text"] = s...
            #sentence["sentiment"] = s...
            sentence["sentiment-scores"] = {
                #"positive": s...,
                #"neutral": s..,
                #"negative": s..
            }

            mined_opinions = []
            for mined_opinion in s.mined_opinions:
                opinion = {}
                #opinion["target-text"] = mined_opinion...
                #opinion["target-sentiment"] = mined_opinion...
                opinion["sentiment-scores"] = {
                    #"positive": mined_opinion...,
                    #"negative": mined_opinion...,
                }
                
                opinion_assessments = []
                for assessment in mined_opinion.assessments:
                    opinion_assessment = {}
                    #opinion_assessment["text"] = assessment...
                    #opinion_assessment["sentiment"] = assessment...
                    opinion_assessment["sentiment-scores"] = {
                        #"positive": assessment...,
                        #"negative": assessment...
                    }
                    opinion_assessments.append(opinion_assessment)

                opinion["assessments"] = opinion_assessments
                mined_opinions.append(opinion)

            sentence["mined_opinions"] = mined_opinions
            sentences.append(sentence)

        sentiment["sentences"] = sentences
    
    #return sentiment
    raise NotImplementedError

def create_named_entity_extraction_request(call_contents):
    # TODO:
    # 1. The call_contents parameter is formatted as a list of strings. Join them together with spaces to pass in as a single document.

    # 2. Create a TextAnalyticsClient, connecting it to your Language Service endpoint.
    #client = ...

    # 3. Recognize entities within the call transcript.
    #result = client...
    # Create named_entity list as a JSON array
    named_entities = []

    # 4. Add each extracted named entity to the named_entity array.
    #for entity in result.entities:
    #    named_entities.append({
    #        "text": ...,
    #        "category": ...,
    #        "subcategory": ...,
    #        "length": ...,
    #        "offset": ...,
    #        "confidence-score": ...
    #    })

    #return named entities
    raise NotImplementedError


def main():
    call_contents = []
    st.write(
    """
    # Call Center

    This Streamlit dashboard is intended to replicate some of the functionality of a call center monitoring solution. It is not intended to be a production-ready application.
    """
    )

    st.write("## Simulate a call")

    uploaded_file = st.file_uploader("Upload an audio file", type="wav")
    if uploaded_file is not None and ('file_transcription' not in st.session_state or st.session_state.file_transcription is False):
        st.audio(uploaded_file, format='audio/wav')
        with st.spinner("Transcribing the call..."):
            all_results = create_transcription_request(uploaded_file, speech_key, speech_region)
            st.session_state.file_transcription_results = all_results
            st.session_state.file_transcription = True
        st.success("Transcription complete!")

    if 'file_transcription_results' in st.session_state:
        st.write(st.session_state.file_transcription_results)

        
    st.write("## Perform a Live Call")

    start_recording = stx.button("Record", key="recording_in_progress")
    if start_recording:
        with st.spinner("Transcribing your conversation..."):
            create_live_transcription_request(speech_key, speech_region)

    if 'transcription_results' in st.session_state:
        st.write(st.session_state.transcription_results)
    else:
        print("Nothing in transcription results!")


    if 'transcription_results' in st.session_state:
        st.write(st.session_state.transcription_results)

    st.write("""## Clear Messages between Calls
        Select this button to clear out session state and refresh the page.
        Do this before loading a new audio file or recording a new call.
        This will ensure that transcription and compliance checks will happen correctly.
    """)

    if st.button("Clear messages"):
        if 'file_transcription_results' in st.session_state:
            del st.session_state.file_transcription_results
        if 'transcription_results' in st.session_state:
            del st.session_state.transcription_results
        streamlit_js_eval(js_expressions="parent.window.location.reload()")

    st.write("## Is Your Call in Compliance?")

    include_recording_message = st.checkbox("Call needs an indicator we are recording it")
    is_relevant_to_topic = st.checkbox("Call is relevant to the hotel and resort industry")

    if st.button("Check for Compliance"):
        with st.spinner("Checking for compliance..."):
            if 'file_transcription_results' in st.session_state:
                call_contents = st.session_state.file_transcription_results
            elif 'transcription_results' in st.session_state:
                call_contents = st.session_state.transcription_results
            else:
                st.write("Please upload an audio file or record a call before checking for compliance.")
            if call_contents is not None and len(call_contents) > 0:
                compliance_results = is_call_in_compliance(call_contents, include_recording_message, is_relevant_to_topic)
                st.write(compliance_results)
        st.success("Compliance check complete!")

    # Exercise 6: Generate call summaries
    st.write("## Generate call summaries")

    if st.button("Generate extractive summary"):
        # TODO: Complete the logic for this button click by doing the following:
        # 1. Set call_contents to file_transcription_results. If it is empty, write out an error message for the user.
        # 2. Use st.spinner() to wrap the summarization process.
        # 3. Call the generate_extractive_summary function and set its results to a variable named extractive_summary.
        # 4. Call st.success() to indicate that the extractive summarization process is complete.
        # 5. Save the extractive_summary value to session state.
        # 6. Write the extractive_summary value to the Streamlit dashboard.
        raise NotImplementedError
    
    if st.button("Generate abstractive summary"):
        # TODO: Complete the logic for this button click by doing the following:
        # 1. Set call_contents to file_transcription_results. If it is empty, write out an error message for the user.
        # 2. Use st.spinner() to wrap the summarization process.
        # 3. Call the generate_abstractive_summary function and set its results to a variable named abstractive_summary.
        # 4. Call st.success() to indicate that the extractive summarization process is complete.
        # 5. Save the abstractive_summary value to session state.
        # 6. Write the abstractive_summary value to the Streamlit dashboard.
        raise NotImplementedError
    
    if st.button("Generate query-based summary"):
        # TODO: Complete the logic for this button click by doing the following:
        # 1. Set call_contents to file_transcription_results. If it is empty, write out an error message for the user.
        # 2. Use st.spinner() to wrap the summarization process.
        # 3. Call generate_query_based_summary function and set its results to a variable named openai_summary.
        # 4. Call st.success() to indicate that the query-based summarization process is complete.
        # 5. Save openai_summary value to session state.
        # 6. Write the openai_summary value to the Streamlit dashboard.
        raise NotImplementedError
    
    st.write("## Analyze call sentiment and perform opinion mining")

    # TODO: Add a Streamlit button to labeled "Analyze sentiment and mine opinions". If the button is clicked:
        # 1. Set call_contents to file_transcription_results. If it is empty, write out an error message for the user.
        # 2. Use st.spinner() to wrap the sentiment analysis process.
        # 3. Call create_sentiment_analysis_and_opinion_mining_request function and set its results to a variable named sentiment_and_mined_opinions.
        # 4. Call st.success() to indicate that the sentiment analysis process process is complete.
        # 5. Save sentiment_and_mined_opinions value to session state.
        # 6. Write the sentiment_and_mined_opinions value to the Streamlit dashboard.

    st.write("## Extract named entities")

    # TODO: Add a Streamlit button to labeled "Extract named entities". If the button is clicked:
        # 1. Set call_contents to file_transcription_results. If it is empty, write out an error message for the user.
        # 2. Use st.spinner() to wrap the named entity extraction process process.
        # 3. Call create_named_entity_extraction_request function and set its results to a variable named named_entities.
        # 4. Call st.success() to indicate that the entity extraction process process is complete.
        # 5. Save named_entities value to session state.
        # 6. Write the named_entities value to the Streamlit dashboard.
    
if __name__ == "__main__":
    main()
