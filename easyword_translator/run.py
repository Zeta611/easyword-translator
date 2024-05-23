import logging
import os
import warnings

import gradio as gr
import pandas as pd
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_upstage import ChatUpstage
from rapidfuzz import process

import pymupdf

warnings.filterwarnings("ignore")


UPSTAGE_API_KEY = os.environ["UPSTAGE_API_KEY"]

TITLE = "쉬운 전문용어 번역기"
DESCRIPTION = """
우리가 easyword.kr에 모은 쉬운 전문용어들을 사용하여, 자동으로 문장을 번역해줍니다.
번역을 자동으로 하기 위해 업스테이지의 Solar 대형 언어 모델 (large language model, LLM)을 사용합니다.
우리 번역기는 최대한 쉬운 전문용어 원칙을 따르며, 원문 전문용어는 번역된 쉬운말 다음 괄호 안에 따라붙입니다.
""".strip()

LIMIT_FACTOR = 2.5
SCORE_CUTOFF = 60.0
MAX_RETRIES = 4

logging.basicConfig(
    filename=f"{__file__}.log",
    filemode="a",
    format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.DEBUG,
)
logger = logging.getLogger(__file__)

df = pd.read_csv("./dictionary.csv")
grouped_df = df.groupby("name").agg(lambda x: ",".join(map(str, x)))
grouped_df = grouped_df.drop(columns=["Unnamed: 0"])


def find_jargons(sentence: str, limit: int | None = None) -> list[str]:
    if limit is None:
        limit = int(len(sentence.split()) * LIMIT_FACTOR)
    extracted = process.extract(
        sentence, grouped_df.index, limit=limit, score_cutoff=SCORE_CUTOFF
    )
    return [v[0] for v in extracted]


def recommend_prompt(jargon: str) -> str:
    return f"'{jargon}'은 '{grouped_df.loc[jargon].name_trans}'로"


llm = ChatUpstage()


def chainer(messages):
    return ChatPromptTemplate.from_messages(messages) | llm | StrOutputParser()


SYSTEM_PROMPT = """
    너는 컴퓨터 과학 및 공학 분야의 전문 용어를 쉬운 우리말로 번역해주는 번역가야.
    전문용어의 의미를 정확히 이해하고, 그 의미가 정확히 전달되는 쉬운말을 찾아야 해.
    지레 겁먹게하는 용어(불필요한 한문투)를 피하고, 가능하면 쉬운말을 찾아야 해.
    원문 전문용어는 해당 우리말 다음에 괄호안에 항상 따라붙여야 해.
    기존의 권위에 얽매이지 않고, 기존 용어사전이나 이미 널리퍼진 용어지만 쉽지않다면, 보다 쉬운 전문용어를 찾아야 해.
    이때, 기존용어는 원문 전문용어와 함께 괄호안에 따라붙여야 해.
    쉬운말은 순수 우리말을 뜻하지 않아. 외래어라도 널리 쉽게 받아들여진다면 사용해.
    """

SAMPLE_SENTENCE = "In functional programming, continuation-passing style (CPS) is a style of programming in which control is passed explicitly in the form of a continuation."
SAMPLE_TRANSLATION = "값중심 프로그래밍[functional programming]에서, 마저할일 전달하기[continuation-passing style, CPS]는 실행흐름[control]이 직접 마저할일[continutaion]의 형태로 전달되는 프로그래밍 스타일이다."


def translate(sentence: str) -> str:
    messages = [
        ("system", SYSTEM_PROMPT),
        (
            "human",
            f"전문 용어를 번역할 때는 반드시 원어를 괄호[]에 넣어서 따라 붙여야 해. 이 문장을 번역해줘: '{SAMPLE_SENTENCE}'",
        ),
        ("ai", SAMPLE_TRANSLATION),
        (
            "human",
            f"전문 용어를 번역할 때는 반드시 원어를 괄호[]에 넣어서 따라 붙여야 해. 이 문장을 번역해줘: '{sentence}'",
        ),
    ]

    initial_translation = chainer(messages).invoke({})
    logger.info(initial_translation)

    used_jargons = find_jargons(sentence)
    messages += [
        ("ai", initial_translation),
        (
            "human",
            f"방금 번역한 문장에서 '{', '.join(used_jargons)}' 중 사용한 용어가 있다면, 어떤 용어들로 번역했는지 말해줘. 사용하지 않은 용어들은 무시해도 돼.",
        ),
    ]
    response = chainer(messages).invoke({})
    logger.info(response)

    recommendations = ", ".join(recommend_prompt(jargon) for jargon in used_jargons)

    messages += [
        ("ai", response),
        (
            "human",
            f"이번에는 처음 번역했던 문장을 '{sentence}'를 다시 번역해주는데, 다음 목록에 나온 쉬운 전문용어 번역 예시를 참고해서 번역을 해줘: '{recommendations}' 사용하지 않은 용어들은 무시해도 돼. 추가 설명 없이 문장만 번역해. 사용된 원어를 용어 바로 뒤에 괄호 []에 넣어서 따라 붙여줘.",
        ),
    ]
    refined_translation = chainer(messages).invoke({})
    logger.info(refined_translation)

    retries = 0
    while "[" not in refined_translation or "]" not in refined_translation:
        retries += 1
        if retries > MAX_RETRIES:
            break
        messages += [
            ("ai", refined_translation),
            (
                "human",
                f"전문용어를 번역했으면 반드시 원어를 괄호[]에 넣어서 따라 붙여야 해. '실행흐름[control]'처럼. 방금 번역한 '{refined_translation}'에서, 원래 문장 '{sentence}'에 사용된 원어를 용어 바로 뒤에 괄호 []에 넣어서 따라 붙여줘.",
            ),
        ]
        try:
            refined_translation = chainer(messages).invoke({})
        except Exception as e:
            logger.error(e)
            break

        logger.info(refined_translation)

    refined_translation = refined_translation.replace("[", "(").replace("]", ")")
    return refined_translation


class PDFFile:
    def __init__(self):
        self.file_list = []

    def read_pdf(self, file_path: str) -> str:
        # Open the PDF file
        document = pymupdf.open(file_path)
        text = ""

        # Iterate through the pages
        for page_num in range(len(document)):
            # Extract text from each page
            page = document.load_page(page_num)
            text += page.get_text()

        # Close the PDF document
        document.close()

        return text

    def remove_line_breaks(self, text: str) -> list[str]:
        #  remove only single line breaks, not paragraphs
        # find line breaks and it is not followed by a period
        for i in range(len(text)):
            if i == 0 or i == len(text) - 1:
                continue
            if text[i] == "\n" and text[i - 1] != "." and text[i + 1] != "\n":
                text = text[:i] + " " + text[i + 1 :]
        return text

    def upload_file(self, file_path: str) -> list[str]:
        self.file_list.append(file_path)
        return self.file_list

    def transalte_pdf(
        self,
        remove_line_breaks: bool,
        save_before_translation: bool,
    ) -> str:
        if not self.file_list:
            return "No file uploaded yet."

        file_out_list = []
        for file in self.file_list:
            directory = os.path.dirname(file)
            filename = os.path.basename(file)

            # remove extension
            filename = ".".join(filename.split(".")[:-1])
            pdf_text = self.read_pdf(file)
            if remove_line_breaks:
                pdf_text = self.remove_line_breaks(pdf_text)

            if save_before_translation:
                with open(f"{directory}/{filename}_pre.txt", "w") as f:
                    f.write(pdf_text)
            file_out_list.append(f"{directory}/{filename}_pre.txt")

            # translation = translate(pdf_text)
            # Translation with divide and conquer with 50 sentences
            translation = ""
            # seperate the text into sentences
            sentences = pdf_text.split(".")
            for i in range(0, len(sentences), 50):
                translation += translate(".".join(sentences[i : i + 50])) + ". "

            with open(f"{directory}/{filename}_translated.txt", "w") as f:
                f.write(translation)
            file_out_list.append(f"{directory}/{filename}_translated.txt")

        self.file_list = []
        # Zip the files
        # import zipfile

        # with zipfile.ZipFile(f"{directory}/translated_files.zip", "w") as z:
        #     for file in file_out_list:
        #         z.write(file)

        # return f"{directory}/translated_files.zip"
        return file_out_list


with gr.Blocks() as demo:
    with gr.Tab("TEXT"):
        chatbot = gr.Interface(
            fn=translate,
            inputs=gr.Textbox(label="Enter your text"),
            outputs=[gr.Textbox(label="Translation")],
            examples=[
                "In functional programming, continuation-passing style (CPS) is a style of programming in which control is passed explicitly in the form of a continuation.",
                "In computer science, abstract interpretation is a theory of sound approximation of the semantics of computer programs, based on monotonic functions over ordered sets, especially lattices.",
                "In computer science, functional programming is a programming paradigm where programs are constructed by applying and composing functions",
                "Lambda calculus (also written as λ-calculus) is a formal system in mathematical logic for expressing computation based on function abstraction and application using variable binding and substitution",
                "Operational semantics is a category of formal programming language semantics in which certain desired properties of a program, such as correctness, safety or security, are verified by constructing proofs from logical statements about its execution and procedures, rather than by attaching mathematical meanings to its terms (denotational semantics).",
                "In computing and computer programming, exception handling is the process of responding to the occurrence of exceptions – anomalous or exceptional conditions requiring special processing – during the execution of a program.",
                "The term redex, short for reducible expression, refers to subterms that can be reduced by one of the reduction rules.",
            ],
            title=TITLE,
            description=DESCRIPTION,
        )
    with gr.Tab("PDF"):
        pdf_file = PDFFile()
        upload_button = gr.UploadButton(
            label="Upload PDF",
            file_types=[".pdf"],
        )
        upload_file_list_box = gr.File(label="Uploaded Files")

        upload_button.upload(pdf_file.upload_file, upload_button, upload_file_list_box)

        run_translator = gr.Interface(
            fn=pdf_file.transalte_pdf,
            inputs=[
                gr.Checkbox(label="Remove line breaks"),
                gr.Checkbox(label="Save before translation"),
            ],
            outputs=[gr.File(label="Download Translated Files")],
        )


def main():
    demo.launch(share=True)


if __name__ == "__main__":
    main()
