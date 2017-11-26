import re
import string


VN_ACCENTS = "aáàảãạăắằẳẵặâấầẩẫậđ₫eéèẻẽẹêếềểễệiíìỉĩị" \
             "oóòỏõọôốồổỗộơớờởỡợuúùủũụưứừửữựyýỳỷỹỵ"

WORDS = ''.join(
    set(VN_ACCENTS) | set(string.ascii_lowercase) | set(string.digits)
)

# punct_re = re.compile('[{}]'.format(re.escape(string.punctuation)))

punct_re = re.compile('[{}]'.format(
    re.escape(string.punctuation + string.whitespace)
))


def preprocess_vietnamese_accent(text):
    """
    Remove all specific charracters not in WORDS
    """
    text_preprocess = re.sub(
        "\s\s+", " ",
        re.sub('[^.{}]'.format(WORDS), ' ', text.lower()).strip()
    )

    return text_preprocess


def preprocess(text):
    text = text or ""
    text = punct_re.sub(" ", text)  # remove punctuation
    text = re.sub("\s\s+", " ", text)  # remove multiple spaces
    return text.strip()


def remove_space(text):
    text = re.sub("\s", "", text)
    return text


def process_raw_documents(description):
    description = preprocess(description)
    raw_description = preprocess_vietnamese_accent(description)
    return raw_description
