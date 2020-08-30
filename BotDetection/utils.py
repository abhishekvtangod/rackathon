import praw, operator, string, csv, numpy as np
from nltk.corpus import stopwords 

np.random.seed(1)

r = praw.Reddit(client_id='SI8pN3DSbt0zor',
                     client_secret='xaxkj7HNh8kwg8e5t4m6KvSrbTI',
                     password='1guiwevlfo00esyy',
                     user_agent='testscript by /u/fakebot3',
                     username='redaniket') #Reddit authentication   

blacklisted = ['askouija', 'test', 'freekarma4you'] #Subreddits where people engage in bot-like behavior


def botInName(user):
    return 1 if 'bot' in str(user).lower() else 0.01

def getSpread(user, comlist):
    '''
    A number between 0 and 100, representing how spread out the user's activity is.
    Bots comment in many subreddits; real users tend to stick to a few.
    '''
    subsList = []
    for item in comlist:
        if str(item.subreddit) not in subsList: 
            subsList.append(str(item.subreddit))
    return 0.01 if float(len(comlist))/float(len(subsList)) == 0 else float(len(comlist))/float(len(subsList))  


def getSameLevels(user, comlist):
    '''
    On reddit, there are top-level comments and also child comments. If an account
    only posts one type of comment, it's probably a bot. This function returns a number
    between 0 and 1 that represents the percentage of comments in an account that are top-level
    or children (replies to other comments), whichever is higher.
    '''
    total = 0
    tlevel = 0
    for comment in comlist:
        total += 1
        if comment.is_root:
            tlevel += 1
    pcount = max(float(tlevel)/float(total), (float(total)-float(tlevel))/float(total))
    return 0.01 if pcount == 0 else pcount 

def uniqueComments(user, comlist):
    '''
    Returns a number between 0 and 1 that represents what percentage of 
    the user's comments are unique.
    '''
    comlist = [x for x in comlist if x.distinguished == None] #Ignore moderator comments, which tend to be repetitive
    x = 0 
    uniqueComments = []
    flist = [j for j in comlist]
    flength = len(flist)
    freqDict = {}
    while x < len(flist):
        content = flist[x].body
        if content.lower() not in uniqueComments:
            uniqueComments.append(content.lower())
        x += 1
    return len(uniqueComments)/len(comlist)


def isRepetitive(user, comlist):
    '''
    Returns a number between 0 and 1 that measures a user's repetitiveness based on
    how many words they repeat in their comments.
    '''
    comlist = [x for x in comlist if x.distinguished == None]
    if len(comlist) < 15:
        return 0.01
    
    x = 0 
    uniqueComments = []
    flist = [j for j in comlist]
    flength = len(flist)
    freqDict = {}
    while x < len(flist):
        aldone = []
        bad = stopwords.words('english') + stopwords.words('spanish') + stopwords.words('portuguese') + list(string.punctuation) #filter out common words
        content = flist[x].body.replace("["," ").replace("]"," ")
        if 'http' in content.lower() or 'www.' in content.lower():
            content = content.replace('/', ' ')
        if content.lower() not in uniqueComments:
            uniqueComments.append(content.lower())
        for key in list(string.punctuation) + ['\n']:
            content = content.replace(key, '')
        words = [i for i in content.lower().split(' ') if i.replace("'s",'').replace("'nt",'').replace("'ve",'').replace("'ll",'').replace("'m",'') not in bad and i.strip("'`").strip('`') != '']
        for thing in words:
            if  thing not in aldone:
                aldone.append(thing)
                try: 
                    freqDict[thing] += 1
                except:
                    freqDict[thing] = 1
            
        x += 1
    oDict = sorted(freqDict.items(), key=operator.itemgetter(1))
    try:
        return (float(oDict[-1][1])/float(flength) + float(oDict[-2][1])/float(flength))/2 
    except IndexError:
        return (float(oDict[-1][1])/float(flength))

def keyWordPres(user, comlist):
    '''
    Is there a keyword (one that bots frequently use in their comments) in many of this user's
    comments? This function returns a number between 0 and 1 that measurs that.
    '''
    comlist = [x for x in comlist if x.distinguished == None]
    tcount = 0
    hopeful = {"bot":0, "source code":0, "feedback":0, "contact":0, "faq":0, "*":0, "**":0, "^":0}
    for comment in comlist:
        tcount += 1
        for item in hopeful:
            if item in comment.body.lower():
                hopeful[item] += 1
    hopeful_sort = sorted(hopeful.items(), key=operator.itemgetter(1))
    return 0.01 if float(hopeful_sort[-1][1])/float(tcount) == 0 else float(hopeful_sort[-1][1])/float(tcount) 

def fewPosts(user, innum):
    '''
    This algorithm was designed specifically to detect comment bots. If a user has very few posts
    relative to their number of comments, they are probably a bot. This function returns a number
    between 0 (no comments, many posts) and 100 (many comments, no posts) that measures that.
    '''
    tplist = [post for post in r.redditor(user).submissions.new(limit=100)]
    return 100.0 if len(tplist) == 0 else float(innum)/float(len(tplist))


def avTime(user, comlist):
    '''
    Average time between comments; bots tend to have very low values for this, since they never take breaks.
    '''
    x = 0 
    totals = []
    flist = [j for j in comlist]
    while x < len(flist)-1:
        totals.append(flist[x].created_utc - flist[x+1].created_utc)
        x+=1
    result = float(sum(totals))/float(len(totals))
    return result 


def timeToResponse(user, comlist):
    ''' 
    Bots respond to posts and comments very quickly. This function returns 
    the average amount of time (in seconds) it takes for that to occur.
    '''
    allTimes = []
    for comment in comlist[:10]:
        allTimes.append(comment.created_utc-comment.parent().created_utc)
    return sum(allTimes)/len(allTimes)
    

class User:
    def __init__(self, usr):
        comment_list = [x for x in r.redditor(usr).comments.new(limit=100) if str(x.subreddit).lower() not in blacklisted]
        if len(comment_list) < 15:
            self.invalid_flag = True 
        else: 
            self.invalid_flag = False 
            self.bot_in_name = botInName(usr)
            self.same_level_comments = getSameLevels(usr, comment_list)
            self.keyword_present = keyWordPres(usr, comment_list)
            self.unique_comments = uniqueComments(usr, comment_list)
            self.few_posts = fewPosts(usr, len(comment_list))
            self.is_repetitive = isRepetitive(usr, comment_list)
            self.spread = getSpread(usr, comment_list)
            self.average_time_beetween_comments = avTime(usr, comment_list)
            self.average_time_to_reply = timeToResponse(usr, comment_list)
            self.data = [self.invalid_flag, self.bot_in_name, self.same_level_comments, self.keyword_present, 
            self.unique_comments, self.few_posts, self.is_repetitive, self.spread, self.average_time_beetween_comments,
            self.average_time_to_reply] #Why make this annoying list? Because SomeClass.__dict__.values() is not necessarily sorted in Python3.5 and lower