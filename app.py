#!/usr/bin/env python3
import os
import pickle
import traceback
from datetime import date

import numpy as np
import openai
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from dotenv import load_dotenv
from fuzzywuzzy import fuzz

# ─── Initialise ──────────────────────────────────────────────────────────────
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise RuntimeError("OPENAI_API_KEY not set in .env")

# ─── PAGE_LINKS ───────────────────────────────────────────────────────────────
PAGE_LINKS = {
    # Home
    "home":                "https://cognitivetasking.com/",
    "homepage":            "https://cognitivetasking.com/",
    "main page":           "https://cognitivetasking.com/",

    # Downloads
    "downloads":           "https://cognitivetasking.com/PENai-downloads",
    "penai downloads":     "https://cognitivetasking.com/PENai-downloads",

    # Benefits calculator
    "benefits calculator": "https://cognitivetasking.com/benefits-calculator",
    "savings":             "https://cognitivetasking.com/benefits-calculator",
    "improvements":        "https://cognitivetasking.com/benefits-calculator",

    # Book a meeting
    "book a meeting":      "https://cognitivetasking.com/book-a-meeting",
    "meeting":             "https://cognitivetasking.com/book-a-meeting",
    "demo":                "https://cognitivetasking.com/book-a-meeting",
    "book":                "https://cognitivetasking.com/book-a-meeting",
    
    # Early adopter programme
    "early adopter programme": "https://cognitivetasking.com/early-adopter-programme",

    # FAQs
    "faq":                 "https://cognitivetasking.com/PENai-FAQs",
    "faqs":                "https://cognitivetasking.com/PENai-FAQs",

    # Contact
    "contact":             "https://cognitivetasking.com/contact-us",
    "contact us":          "https://cognitivetasking.com/contact-us",
    "email":               "https://cognitivetasking.com/contact-us",
    "phone":               "https://cognitivetasking.com/contact-us",
    "telephone":           "https://cognitivetasking.com/contact-us",
}


# ─── Human labels for fallback links ────────────────────────────────────────
# Home
    PAGE_LINKS["home"]:        "Back to Cognitive Tasking home",

    # Downloads
    PAGE_LINKS["downloads"]:   "View downloads",

    # Benefits calculator
    PAGE_LINKS["benefits calculator"]: "Open benefits calculator",

    # Book a meeting
    PAGE_LINKS["book a meeting"]:      "Book a meeting",

    # Early adopter programme
    PAGE_LINKS["early adopter programme"]: "Learn about Early Adopter Programme",

    # FAQs
    PAGE_LINKS["faq"]:         "View FAQs",

    # Contact
    PAGE_LINKS["contact"]:     "Contact us",
}

STATIC_QAS = {
       # ─── PEN.ai Overview ──────────────────────────────────────────────
    "what is pen.ai": (
        "PEN.ai is an AI-powered admissions platform designed to streamline and personalise the admissions process for schools. "
        "It automates administrative tasks, enhances parent and student engagement, and provides actionable insights to admissions teams.",
        PAGE_LINKS["home"],
        "Learn more about PEN.ai"
    ),

    # ─── Differentiators ──────────────────────────────────────────────
    "how does pen.ai differ from other automation tools": (
        "PEN.ai stands out by:\n"
        "- Providing fully personalised admissions experiences\n"
        "- Using sentiment analysis to proactively identify concerns and priorities\n"
        "- Delivering data-driven insights to optimise outreach strategies and family engagement",
        PAGE_LINKS["home"],
        "Learn more about PEN.ai"
    ),

    # ─── Personalised Prospectuses ────────────────────────────────────
    "how does pen.ai create personalised prospectuses": (
        "PEN.ai uses data collected through the enquiry form—such as a child’s interests and aspirations—to generate tailored digital prospectuses. "
        "For example, families interested in STEM will see content featuring robotics labs, coding clubs and science fairs.",
        PAGE_LINKS["home"],
        "Learn more about PEN.ai"
    ),

    # ─── Engagement Tracking ───────────────────────────────────────────
    "how does pen.ai track family engagement": (
        "PEN.ai measures engagement by:\n"
        "- Tracking how often a digital prospectus is accessed\n"
        "- Using sentiment analysis to monitor communication tone and interest levels\n"
        "- Providing insights on enquiry-to-application conversions, event attendance and feedback",
        PAGE_LINKS["home"],
        "Learn more about PEN.ai"
    ),

    # ─── Open Day Enhancements ────────────────────────────────────────
    "how does pen.ai improve open day experiences": (
        "PEN.ai enhances open days by:\n"
        "- Sending detailed logistical information and reminders in advance\n"
        "- Providing customised agendas so families see what matters most to them\n"
        "- Collecting and analysing feedback to improve future events",
        PAGE_LINKS["home"],
        "Learn more about PEN.ai"
    ),

    # ─── Enrolment Rates ──────────────────────────────────────────────
    "how does pen.ai improve enrolment rates": (
        "PEN.ai combines personalisation, sentiment analysis and engagement tracking to:\n"
        "- Ensure families feel valued and understood throughout the admissions journey\n"
        "- Prioritise high-potential families for follow-up\n"
        "- Optimise communication to improve conversion rates",
        PAGE_LINKS["home"],
        "Learn more about PEN.ai"
    ),

    # ─── Seasonal Peaks ───────────────────────────────────────────────
    "can pen.ai handle seasonal peaks in admissions": (
        "Yes. PEN.ai automates enquiry responses, reminders and follow-ups, ensuring efficient management of high volumes during open days or admissions deadlines.",
        PAGE_LINKS["home"],
        "Learn more about PEN.ai"
    ),

    # ─── Scalability ─────────────────────────────────────────────────
    "is pen.ai scalable for schools of different sizes": (
        "Absolutely. PEN.ai is designed to scale for schools of all sizes—from smaller independent schools to large institutions—ensuring tailored solutions to meet specific needs.",
        PAGE_LINKS["home"],
        "Learn more about PEN.ai"
    ),

    # ─── Integrations ─────────────────────────────────────────────────
    "how does pen.ai integrate with existing systems": (
        "PEN.ai integrates seamlessly with:\n"
        "- CRMs such as Salesforce or HubSpot via API\n"
        "- Email applications such as Outlook or Gmail\n"
        "- Event management platforms such as Eventbrite",
        PAGE_LINKS["home"],
        "Learn more about PEN.ai"
    ),

    # ─── Security & Compliance ────────────────────────────────────────
    "how secure is pen.ai and is it gDPR compliant": (
        "PEN.ai complies with all GDPR and UK data protection laws. It ensures:\n"
        "- All data is securely stored and encrypted\n"
        "- Full transparency through audit trails\n"
        "- Schools maintain control of their data at all times",
        PAGE_LINKS["home"],
        "Learn more about PEN.ai"
    ),

    # ─── Getting Started ──────────────────────────────────────────────
    "how do i get started with pen.ai": (
        "You can get started by:\n"
        "- Requesting a demo for a personalised walkthrough\n"
        "- Booking a consultation to discuss your school’s unique needs\n"
        "- Onboarding, where we guide you through implementation and training",
        PAGE_LINKS["home"],
        "Learn more about PEN.ai"
    ),
    
# ─── Contact ──────────────────────────────────────────────
    "contact": (
        "You can reach the Cognitive Tasking team at hello@cognitivetasking.com or via our contact page:",
        PAGE_LINKS["contact"],
        "Contact us"
    ),
    "contact us": (
        "You can reach the Cognitive Tasking team at hello@cognitivetasking.com or via our contact page:",
        PAGE_LINKS["contact"],
        "Contact us"
    ),
    "how can i contact cognitive tasking": (
        "You can reach the Cognitive Tasking team at hello@cognitivetasking.com or via our contact page:",
        PAGE_LINKS["contact"],
        "Contact us"
    ),
    
    # ─── Downloads ──────────────────────────────────────────────
    "downloads": (
        "You can find all our PEN.ai downloads and resources here:",
        PAGE_LINKS["downloads"],
        "View downloads"
    ),
    "penai downloads": (
        "You can find all our PEN.ai downloads and resources here:",
        PAGE_LINKS["downloads"],
        "View downloads"
    ),
    "where do i download": (
        "You can find all our PEN.ai downloads and resources here:",
        PAGE_LINKS["downloads"],
        "View downloads"
    ),

    # ─── Benefits Calculator ──────────────────────────────────
    "benefits calculator": (
        "Use our benefits calculator to estimate potential value and savings:",
        PAGE_LINKS["benefits calculator"],
        "Open benefits calculator"
    ),
    "calculator": (
        "Use our benefits calculator to estimate potential value and savings:",
        PAGE_LINKS["benefits calculator"],
        "Open benefits calculator"
    ),

    # ─── Book a Meeting ───────────────────────────────────────
    "book a meeting": (
        "To schedule a meeting with our team, please use the link below:",
        PAGE_LINKS["book a meeting"],
        "Book a meeting"
    ),
    "schedule a meeting": (
        "To schedule a meeting with our team, please use the link below:",
        PAGE_LINKS["book a meeting"],
        "Book a meeting"
    ),

    # ─── Early Adopter Programme ──────────────────────────────
    "early adopter programme": (
        "Join our Early Adopter Programme for exclusive access to new PEN.ai features and support:",
        PAGE_LINKS["early adopter programme"],
        "Learn about Early Adopter Programme"
    ),

    # ─── FAQs ─────────────────────────────────────────────────
    "faq": (
        "You can find answers to frequently asked questions here:",
        PAGE_LINKS["faq"],
        "View FAQs"
    ),
    "faqs": (
        "You can find answers to frequently asked questions here:",
        PAGE_LINKS["faq"],
        "View FAQs"
    ),
    "frequently asked questions": (
        "You can find answers to frequently asked questions here:",
        PAGE_LINKS["faq"],
        "View FAQs"
    ),
}

# ─── Load embeddings & metadata ───────────────────────────────────────────────
with open("embeddings.pkl", "rb") as f:
    embeddings = np.stack(pickle.load(f), axis=0)
with open("metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

EMB_MODEL  = "text-embedding-3-small"
CHAT_MODEL = "gpt-3.5-turbo"

# ─── System prompt ────────────────────────────────────────────────────────────
today = date.today().isoformat()
system_prompt = (
    f"You are a friendly, professional assistant for Cognitive Tasking.\n"
    f"Today's date is {today}.\n"
    "Begin with 'Thank you for your question!' and end with 'Anything else I can help you with today?'.\n"
    "If you do not know the answer, say 'I'm sorry, I don't have that information.'\n"
    "Use British spelling."
)

app = Flask(__name__)
CORS(app, resources={r"/ask": {"origins": "*"}})


def cosine_similarities(matrix, vector):
    dot = matrix @ vector
    norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(vector)
    return dot / (norms + 1e-8)


def remove_bullets(text):
    return " ".join(
        line[2:].strip() if line.startswith("- ") else line.strip()
        for line in text.split("\n")
    )


def format_response(ans):
    footer = "Anything else I can help you with today?"
    ans = ans.replace(footer, "").strip()
    sents, paras, curr = ans.split(". "), [], []
    for s in sents:
        s = s.strip()
        if not s:
            continue
        curr.append(s.rstrip("."))
        if len(curr) >= 3 or s.endswith("?"):
            paras.append(". ".join(curr) + ".")
            curr = []
    if curr:
        paras.append(". ".join(curr) + ".")
    if not paras or not paras[0].startswith("Thank you for your question"):
        paras.insert(0, "Thank you for your question!")
    paras.append(footer)
    return "\n\n".join(paras)


@app.route("/", methods=["GET"])
def home():
    return "PEN.ai is running."


@app.route("/ask", methods=["POST"])
@cross_origin()
def ask():
    try:
        data = request.get_json(force=True)
        question = data.get("question", "").strip()
        if not question:
            return jsonify(error="No question provided"), 400

        key = question.lower().rstrip("?")

        # 1) Exact static
        if key in STATIC_QAS:
            raw, url, label = STATIC_QAS[key]
            return jsonify(
                answer=format_response(remove_bullets(raw)),
                url=url,
                link_label=label
            ), 200

        # 2) Fuzzy static
        for sk, (raw, url, label) in STATIC_QAS.items():
            if fuzz.partial_ratio(sk, key) > 80:
                return jsonify(
                    answer=format_response(remove_bullets(raw)),
                    url=url,
                    link_label=label
                ), 200

        # 3) Welcome (custom — no “Thank you for your question!”)
        if question == "__welcome__":
            raw = (
                "Hi there! Ask me anything about More House School.\n\n"
                "We tailor our prospectus to your enquiry. For more details, visit below.\n\n"
                "Anything else I can help you with today?"
            )
            return jsonify(
                answer=remove_bullets(raw),
                url=PAGE_LINKS["enquiry"],
                link_label="Enquire now"
            ), 200

        # 4) Guard “how many…”
        if key.startswith("how many"):
            return jsonify(
                answer=format_response("I'm sorry, I don't have that information."),
                url=None
            ), 200

        # 5) Keyword → URL
        relevant_url = None
        for k, u in PAGE_LINKS.items():
            if k in key or any(
                fuzz.partial_ratio(k, w) > 80 for w in key.split() if len(w) > 3
            ):
                relevant_url = u
                break

        # 6) RAG fallback
        emb = openai.embeddings.create(model=EMB_MODEL, input=question)
        q_vec = np.array(emb.data[0].embedding, dtype="float32")
        sims = cosine_similarities(embeddings, q_vec)
        top_idxs = sims.argsort()[-20:][::-1]
        contexts = [metadata[i]["text"] for i in top_idxs]
        prompt = "Use these passages:\n\n" + "\n---\n".join(contexts)
        prompt += f"\n\nQuestion: {question}\nAnswer:"
        chat = openai.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        raw = chat.choices[0].message.content
        answer = format_response(remove_bullets(raw))

        # 7) Fallback URL + human label
        if not relevant_url and top_idxs.size:
            relevant_url = metadata[top_idxs[0]].get("url")

        link_label = URL_LABELS.get(relevant_url)

        return jsonify(
            answer=answer,
            url=relevant_url,
            link_label=link_label
        ), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify(error=str(e)), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
