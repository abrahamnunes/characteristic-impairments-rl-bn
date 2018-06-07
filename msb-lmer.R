library(reshape2)
library(lmerTest)
library(nloptr)

plotdir = 'plots/'

# INSTRUCTIONS
# 	This script in effect performs the convolutional logistic regression analysis,
#	but with the added strength of regularization offered through mixed effects modeling.
#
#	The input is a dataframe with 10x201 trials (i.e. 10 steps back for each trial) and
#	the following columns 
#		y = {0=choice A, 1=choice B} at trial t 
#		kback = {1,...,10} indexing the number of trials back represented by {RC,RU,OC,OU}
#		RC = {0=if trial t-kback was not Rewarded & common transition, 
#			  1=if trial t-kback was Rewarded & Common transition + Choice B selected, 
#			  -1=if trial t-kback was Rewarded & Common transition + Choice A selected}
#		RU = {Analogous to RC, but Rewarded + Uncommon transition}
#		OC = {Analogous to RC, but omitted + common transition}
#		OU = {Analogous to RC, but omitted + uncommon transition}
#		subject_id = {unique factor variable identifying each subject}
#
# 	Note here we regress the Trialtype:kback rather than Trialtype*kback. This achieves the 
#	same form as the convolutional model (think of representing the convolution as a matrix 
#	multiplication). 

data = 'PUT YOUR DATAFRAME HERE'

# Mixed effects model 

fm = glmer(y ~ (RC + RU + OC + OU):kback + ((RC + RU + OC + OU):kback + 1 | subject_id), 
           data=data,
           family=binomial(link='logit'),
           nAGQ=0,
           glmerControl(optimizer='nloptwrap', 
                        calc.derivs=FALSE,
                        optCtrl=list(maxfun=300000)))
summary(fm)

df = coef(fm)$subject_id
write.csv(df, 'msb_results_raw.csv')
df$subject_id = row.names(df)
df = melt(data=df, id.vars = c('subject_id'))
df$type = NA
df$type[grep("RC", df$variable)] = "RC"
df$type[grep("RU", df$variable)] = "RU"
df$type[grep("OC", df$variable)] = "OC"
df$type[grep("OU", df$variable)] = "OU"
df$kback = df$variable
df$kback = sub('RC', '', sub('kback', '', sub(':', '', df$kback)))
df$kback = sub('RU', '', sub('kback', '', sub(':', '', df$kback)))
df$kback = sub('OC', '', sub('kback', '', sub(':', '', df$kback)))
df$kback = sub('OU', '', sub('kback', '', sub(':', '', df$kback)))
write.csv(df, 'msb_results.csv')
