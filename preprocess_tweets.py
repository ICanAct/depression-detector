import re
from emoji import demojize
from nltk.tokenize import TweetTokenizer
import string
import inflect

tokenizer = TweetTokenizer()

general_hashtags = ['#mentalhealth', '#anxiety', '#depression', '#bipolar', '#ptsd', '#anorexia', '#bulimia']
#anxiety
anxiety_hashtags = ['#anxietyhelp', '#anxietyattack', '#anxietydisorder', '#anxietysupport', '#fuckanxiety', '#anxietyawareness']
#depression
depression_hashtags = ['#depressionhelp', '#fuckdepression', '#depresion', '#depressionrecovery']
#bipolar
bipolar_hashtags = ['#bipolarhelp', '#bipolarawareness', '#bipolardisorder', '#bipolarrecovery', '#bippolar', '#bipolar1', 'bipolar2', '#bpd']
#ptsd
ptsd_hashtags = ['#ptsdhelp', '#ptsdawareness', '#ptsdrecovery', '#ptsdsupport', '#ptsdtherapy', '#cptsd', '#fuckptsd', 'ptsdsucks']
#eating disorders
eating_disorders_hashtags = ['#eatingdisorder', '#eatingdisorderhelp', '#anorexiahelp', '#bulimiahelp', '#ednos', '#ednoshelp', '#edrecovery', '#edawareness']
#amalgamated hashtags
mental_health_hashtags = general_hashtags + anxiety_hashtags + depression_hashtags + bipolar_hashtags + ptsd_hashtags + eating_disorders_hashtags


def normalizeToken(token):
    #remove non-ascii characters
    token = token.encode('ascii', errors='ignore').decode()
    #handle ' before it is removed by punctuation
    token = re.sub('\'ll', ' will', token)
    token = re.sub('\'ve', ' have', token)
    #remove 2 or more spaces with 1 space
    token = re.sub('\s+', ' ', token)
    #remove punctuation except # and @
    remove_punc = string.punctuation.replace("#", "")
    remove_punc = remove_punc.replace("@", "")
    token = token.translate(str.maketrans('', '', remove_punc))
    #remove rt (retweet) - removing it here because it may not be twitter specific
    token = re.sub(r'^RT$', ' ', token)
    #normalize @user
    if token.startswith("@"):
        return " "
        #normalize URL
    elif token.startswith("http") or token.startswith("www"):
        return ""
        #detect numbers (but it will translate time to text too 2:45pm->two hundred and forty-five pm)
    elif(token.isdigit()):
        inf_eng = inflect.engine()
        return inf_eng.number_to_words(token)
        #normalize emojis
    elif len(token) == 1:
        return demojize(token)
    else:
        if token == "’":
            return "'"
        elif token == "…":
            return "..."
        else:
            #lowercase
            token = token.lower()
            return token


def normalizeTweet(tweet):
    #remove dates before tokenizing (tokenizer cannot recognize date format and will split it up)
    tweet = re.sub(r'\d{1,2}\/\d{1,2}\/\d{2,4}', ' ', tweet)
    #replace punctuations that are encoded differently, and TLDR which tokenizer will split it up
    tokens = tokenizer.tokenize(tweet.replace("’", "'").replace("…", "...").replace("â€™", "'").replace("TL;DR", "").replace("tl;dr", "'"))
    normTweet = " ".join([normalizeToken(token) for token in tokens])

    #remove all queried hashtags
    for hashtag in mental_health_hashtags:
        normTweet = normTweet.replace(hashtag, "")
    normTweet = (
        normTweet.replace("cannot ", "can not ")
        .replace("n't ", " n't ")
        .replace("n 't ", " n't ")
        .replace("ca n't", "can't")
        .replace("ai n't", "ain't")
    )
    normTweet = (
        normTweet.replace("'m ", " 'm ")
        .replace("'re ", " 're ")
        .replace("'s ", " 's ")
        .replace("'ll ", " 'll ")
        .replace("'d ", " 'd ")
        .replace("'ve ", " 've ")
    )
    normTweet = (
        normTweet.replace(" p . m .", "  p.m.")
        .replace(" p . m ", " p.m ")
        .replace(" a . m .", " a.m.")
        .replace(" a . m ", " a.m ")
    )
    normTweet = (
        normTweet.replace("\n", " ")
        .replace("tldr", "")
        .replace("#", "")
    )
    return " ".join(normTweet.split())



if __name__ == "__main__":
    print(
        normalizeTweet(
            "RT @testuser TL;DR tldr I donâ€™t don't want and I'll can't SC has first 2 presumptive cases... .. of coronavirus, DHEC confirms https://postandcourier.com/health/covid19/sc-has-first-two-presumptive-cases-of-coronavirus-dhec-confirms/article_bddfe4ae-5fd3-11ea-9ce4-5f495366cee6.html?utm_medium=social&utm_source=twitter&utm_campaign=user-share… via @postandcourier on 18/2/23 #anxiety #yahoo"
        )
    )