require(fpp)
data(books)
plot(books)

# Creating separate datasets, for hardcover and paperback
paperback = books[,"Paperback"]
hardcover = books[,"Hardcover"]

plot(paperback)
plot(hardcover)

# Forecasting using Holt's method
#Paperback
forecast_paper <- holt(paperback, h=4)
plot(forecast_paper)
forecast_paper # Forecast 207,208,209,210
summary(forecast_paper) # RMSE 31.66184

#Hardcover
forecast_hard <- holt(hardcover, h=4)
plot(forecast_hard)
forecast_hard # Forecast 247, 250, 253, 256
summary(forecast_hard) # RMSE: 27.43588

# Forecasting using Ses method
#Paperback
ses <- ets(paperback)
summary(ses) #RMSE 33.63769
ses.pred <- forecast(ses, h=4)
plot(ses.pred)
ses.pred # Forecast = 207 for all 4 days

#Hardcover
ses <- ets(hardcover)
summary(ses) #RMSE 27.20031
ses.pred <- forecast(ses, h=4)
plot(ses.pred)
ses.pred # Forecast = 251, 254,257,260

data(books)
plot(books, main = "Data set books")
alpha = seq(0.01, 0.99, 0.01)
SSE = NA
for(i in seq_along(alpha)) {
  fcast = ses(books[,"Paperback"], alpha = alpha[i], initial = "simple")
  SSE[i] = sum((books[,"Paperback"] - fcast$fitted)^2)
}
plot(alpha, SSE, type = "l")
fcastPaperSimple = ses(books[,"Paperback"],
                       initial = "simple",
                       h = 4)
fcastPaperSimple$model$par[1]
plot(fcastPaperSimple)

fcastPaperOpt = ses(books[,"Hardcover"],
                    initial = "optimal",
                    h = 4)
fcastPaperOpt$model$par[1]
plot(fcastPaperOpt)
as.numeric((fcastPaperOpt$mean -
              fcastPaperSimple$mean)/fcastPaperSimple$mean) * 100

