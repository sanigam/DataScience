require(fpp)
data(books)
plot(books)

# Creating separate datasets, for hardcover and paperback
paperback = books[,"Paperback"]
hardcover = books[,"Hardcover"]

plot(paperback)
plot(hardcover)

########Forecasting using Holt's method######

#Paperback
forecast_paper <- holt(paperback, h=4)
plot(forecast_paper)
forecast_paper # Forecast 207,208,209,210
summary(forecast_paper) # RMSE 31.66184
accuracy(forecast_paper)
err <- residuals(forecast_paper )
SSE = sum(err^2) # 30074.17
SSE

#Hardcover
forecast_hard <- holt(hardcover, h=4)
plot(forecast_hard)
forecast_hard # Forecast 247, 250, 253, 256
summary(forecast_hard) # RMSE: 27.43588
err <- residuals(forecast_hard )
SSE = sum(err^2) # 22581.83
SSE


########Forecasting using SES method########
#Paperback
ses <- ses(paperback)
summary(ses) #RMSE 33.63769
ses.pred <- forecast(ses, h=4)
plot(ses.pred)
ses.pred # Forecast = 207,207,207,207
accuracy(ses)
err <- residuals(ses)
SSE = sum(err^2) # 33944.82
SSE

#Hardcover
ses <- ses(hardcover)
summary(ses) #RMSE 31.93101
ses.pred <- forecast(ses, h=4)
plot(ses.pred)
ses.pred # Forecast = 239, 239, 239, 239
accuracy(ses)
err <- residuals(ses)
SSE = sum(err^2) #  30587.69
SSE

#############End of R Code ##########

