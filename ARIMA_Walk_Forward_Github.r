library(readr)     
library(dplyr)    
library(forecast)   
library(tseries)   
library(ranger)     
install.packages(ranger)

data <- read_csv("C:/Users/Rey Leonard Garcia/Downloads/Climate-Data (1).csv",
                 col_types = cols(date = col_character(), .default = col_guess())) %>%
  mutate(date = as.Date(date, format = "%d/%m/%Y")) %>%
  select(date, adm4_pcode, heat_index, solar_rad, wind_speed, uv_rad)

vars <- c("heat_index", "solar_rad", "wind_speed", "uv_rad")

clean_data <- data %>%
  mutate(week = date + (7 - as.integer(format(date, "%u")))) %>%
  group_by(week) %>%
  summarise(across(all_of(vars), ~ mean(.x, na.rm = TRUE)), .groups = "drop") %>%
  arrange(week) %>%
  filter(if_all(all_of(vars), ~ !is.na(.x)))            # .dropna()

weeks             <- clean_data$week
heat_index_weekly <- clean_data$heat_index


train_end  <- as.Date("2019-12-29")
test_start <- as.Date("2020-01-05")


heat_train_ts <- ts(heat_index_weekly[weeks <= train_end], frequency = 52)
heat_test     <- heat_index_weekly[weeks >= test_start]
dates_test    <- weeks[weeks >= test_start]


exo_vars <- c("solar_rad", "wind_speed", "uv_rad")

exo_features <- clean_data %>%
  transmute(week,
            across(all_of(exo_vars), ~ dplyr::lag(.x, 52), .names = "{.col}_lag52"),
            across(all_of(exo_vars), ~ dplyr::lag(.x, 1),  .names = "{.col}_lag1")) %>%
  filter(if_all(-week, ~ !is.na(.x)))                   # .dropna()

exo_train <- filter(exo_features, week <= train_end)
exo_test  <- filter(exo_features, week >= test_start)

x_train <- as.matrix(select(exo_train, -week))
x_test  <- as.matrix(select(exo_test,  -week))


sarimax_train <- ts(heat_index_weekly[match(exo_train$week, weeks)], frequency = 52)


heat_D       <- diff(heat_index_weekly, lag = 52)       # (y - y.shift(52)).dropna()
heat_D_dates <- weeks[-seq_len(52)]


ADF <- function(x, dates) {
  
  adf <- suppressWarnings(tseries::adf.test(x))
  kp  <- suppressWarnings(tseries::kpss.test(x, null = "Level"))
  a_p <- adf$p.value
  k_p <- kp$p.value
  
  if (a_p < 0.05) {
    cat(sprintf("Augmented Dickey-Fuller:\n p-value: %s\n Sample Size: %d\n",
                format(a_p), length(x)))
    cat("Passed ADF\n")
  }
  if (k_p > 0.05) {
    cat(sprintf("KPSS Test:\n p-value: %s\n", format(k_p)))
    cat("Passed KPSS\n")
  }
  if (a_p < 0.05 && k_p > 0.05) {
    cat("The data is Stationary!\n")
    plot(dates, x, type = "l", col = "#800000",
         xlab = "Year", ylab = "Heat Index (D = 1)", main = "Time Series of Heat Index")
  }
  invisible(list(adf = adf, kpss = kp))
}


ARIMA_Search <- function(y) {
  fit <- auto.arima(y,
                    start.p = 0, max.p = 2, start.q = 0, max.q = 2,
                    seasonal = FALSE, stepwise = TRUE, trace = TRUE,
                    approximation = FALSE, ic = "aic")
  print(summary(fit))
  invisible(fit)
}


SARIMA_Search <- function(y, xreg = NULL) {
  fit <- auto.arima(y, xreg = xreg,
                    seasonal = TRUE, D = 1,
                    start.p = 0, max.p = 2, start.q = 0, max.q = 2,
                    start.P = 0, max.P = 1, start.Q = 0, max.Q = 1,
                    stepwise = TRUE, trace = TRUE,
                    approximation = FALSE, ic = "aic")
  print(summary(fit))
  invisible(fit)
}

SARIMAX_Search <- function(y, x) SARIMA_Search(y, xreg = x)


forecast_metrics <- function(actual, pred) {
  err <- pred - actual
  c(MAE         = mean(abs(err)),
    Bias        = mean(err),
    RMSE        = sqrt(mean(err^2)),
    `Max Error` = max(abs(err)),
    `R Squared` = 1 - sum(err^2) / sum((actual - mean(actual))^2))   
}

report_metrics <- function(label, m, run_time) {
  for (nm in names(m)) {
    cat(sprintf("%s for %s: %s\n", nm, label, format(m[[nm]], digits = 7)))
  }
  cat(sprintf("Run Time for %s: %.3f sec\n", label, run_time))
  cat("------------------------------------------\n")
}

plot_forecast <- function(dates, actual, pred, title, pred_label) {
  df   <- data.frame(date = dates, actual = actual, pred = pred)
  cols <- c("Actual (True)" = "black")
  cols[pred_label] <- "#d9534f"
  p <- ggplot(df, aes(x = date)) +
    geom_line(aes(y = actual, colour = "Actual (True)"), linewidth = 0.8, alpha = 0.8) +
    geom_line(aes(y = pred,   colour = pred_label), linewidth = 1, linetype = "dashed") +
    scale_colour_manual(values = cols, name = NULL) +
    labs(title = title, x = "Date", y = "Heat Index (\u00B0C)") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  print(p)
  invisible(p)
}

now <- function() proc.time()[["elapsed"]]


fit_sarima <- function(y, xreg = NULL,
                       order = c(1, 0, 2),
                       seasonal = list(order = c(1, 1, 0), period = 52)) {
  stopifnot(!anyNA(y))
  
  attempts <- list(
    list(method = "CSS-ML"),                       # R default, most stable
    list(method = "ML", transform.pars = TRUE),
    list(method = "CSS")                           # last resort
  )
  
  for (a in attempts) {
    fit <- tryCatch(
      do.call(Arima, c(list(y, order = order, seasonal = seasonal, xreg = xreg,
                            optim.control = list(maxit = 1000)), a)),
      error = function(e) {
        message(sprintf("Arima() failed with method = %s: %s",
                        a$method, conditionMessage(e)))
        NULL
      })
    if (!is.null(fit)) {
      message("Fitted with method = ", a$method)
      return(fit)
    }
  }
  stop("All SARIMA estimation methods failed.")
}

SARIMA_Forecast <- function(y_train, y_test, dates_test) {
  t0   <- now()
  fit  <- fit_sarima(y_train)
  pred <- as.numeric(forecast(fit, h = length(y_test))$mean)
  run_time <- now() - t0
  
  report_metrics("SARIMA(1,0,2)(1,1,0)[52]", forecast_metrics(y_test, pred), run_time)
  plot_forecast(dates_test, y_test, pred,
                "SARIMA Train-Test Cross Validation", "SARIMA Predicted")
}

SARIMA_Walk_Forward <- function(y_train, y_test, dates_test) {
  t0  <- now()
  fit <- fit_sarima(y_train)
  
  full  <- ts(c(as.numeric(y_train), y_test), frequency = 52)
  refit <- Arima(full, model = fit)
  pred  <- tail(as.numeric(fitted(refit)), length(y_test))
  run_time <- now() - t0
  
  report_metrics("SARIMA Walk Forward", forecast_metrics(y_test, pred), run_time)
  plot_forecast(dates_test, y_test, pred,
                "SARIMA Walk Forward Prediction", "SARIMA Walk-Forward Predicted")
}


fit_sarima <- function(y, xreg = NULL,
                       order = c(1, 0, 2),
                       seasonal = list(order = c(1, 1, 0), period = 52)) {
  stopifnot(!anyNA(y))
  
  attempts <- list(
    list(method = "CSS-ML"),                       # R default, most stable
    list(method = "ML", transform.pars = TRUE),
    list(method = "CSS")                           # last resort
  )
  
  for (a in attempts) {
    fit <- tryCatch(
      do.call(Arima, c(list(y, order = order, seasonal = seasonal, xreg = xreg,
                            optim.control = list(maxit = 1000)), a)),
      error = function(e) {
        message(sprintf("Arima() failed with method = %s: %s",
                        a$method, conditionMessage(e)))
        NULL
      })
    if (!is.null(fit)) {
      message("Fitted with method = ", a$method)
      return(fit)
    }
  }
  stop("All SARIMA estimation methods failed.")
}

SARIMA_Forecast <- function(y_train, y_test, dates_test) {
  t0   <- now()
  fit  <- fit_sarima(y_train)
  pred <- as.numeric(forecast(fit, h = length(y_test))$mean)
  run_time <- now() - t0
  
  report_metrics("SARIMA(1,0,2)(1,1,0)[52]", forecast_metrics(y_test, pred), run_time)
  plot_forecast(dates_test, y_test, pred,
                "SARIMA Train-Test Cross Validation", "SARIMA Predicted")
}

SARIMA_Walk_Forward <- function(y_train, y_test, dates_test) {
  t0  <- now()
  fit <- fit_sarima(y_train)
  
  full  <- ts(c(as.numeric(y_train), y_test), frequency = 52)
  refit <- Arima(full, model = fit)
  pred  <- tail(as.numeric(fitted(refit)), length(y_test))
  run_time <- now() - t0
  
  report_metrics("SARIMA Walk Forward", forecast_metrics(y_test, pred), run_time)
  plot_forecast(dates_test, y_test, pred,
                "SARIMA Walk Forward Prediction", "SARIMA Walk-Forward Predicted")
}



prep_xreg <- function(x_train, x_test, cor_cut = 0.95) {
  
  keep <- colnames(x_train)
  cm   <- abs(cor(x_train)); cm[lower.tri(cm, diag = TRUE)] <- 0
  drop <- colnames(x_train)[apply(cm, 2, max) > cor_cut]
  keep <- setdiff(keep, drop)
  if (length(drop)) message("Dropped collinear columns: ", paste(drop, collapse = ", "))
  
  xt <- x_train[, keep, drop = FALSE]
  xs <- x_test[,  keep, drop = FALSE]
  m  <- colMeans(xt); s <- apply(xt, 2, sd)
  list(train = scale(xt, center = m, scale = s),
       test  = scale(xs, center = m, scale = s),   
       cols  = keep)
}

fit_sarimax <- function(y, xreg,
                        order = c(1, 0, 2),
                        seasonal = list(order = c(1, 1, 0), period = 52)) {
  stopifnot(!anyNA(y), !anyNA(xreg), length(y) > 3 * seasonal$period)
  
  attempts <- list(
    list(method = "CSS",    optim.method = "Nelder-Mead"),
    list(method = "CSS-ML", optim.method = "Nelder-Mead"),
    list(method = "CSS-ML", optim.method = "BFGS", transform.pars = TRUE),
    list(method = "CSS")
  )
  for (a in attempts) {
    fit <- tryCatch(
      do.call(Arima, c(list(y, order = order, seasonal = seasonal, xreg = xreg,
                            optim.control = list(maxit = 2000)), a)),
      error = function(e) {
        message(sprintf("Failed [%s / %s]: %s", a$method,
                        if (is.null(a$optim.method)) "default" else a$optim.method,
                        conditionMessage(e)))
        NULL
      })
    if (!is.null(fit)) { message("Fitted with method = ", a$method); return(fit) }
  }
  stop("All SARIMAX estimation methods failed.")
}

SARIMAX_Forecast <- function(y_train, x_train, y_test, x_test, dates_test) {
  stopifnot(length(y_test) == nrow(x_test))
  t0 <- now()
  
  xp <- prep_xreg(x_train, x_test)
  mu <- mean(y_train); s <- sd(y_train)
  y_sc <- ts((as.numeric(y_train) - mu) / s, frequency = 52)
  
  fit  <- fit_sarimax(y_sc, xp$train)
  pred <- as.numeric(forecast(fit, xreg = xp$test, h = nrow(xp$test))$mean) * s + mu
  run_time <- now() - t0
  
  report_metrics("SARIMAX(1,0,2)(1,1,0)[52]", forecast_metrics(y_test, pred), run_time)
  plot_forecast(dates_test, y_test, pred,
                "SARIMAX Train-Test Cross Validation", "SARIMAX Predicted")
}




ARIMA_Forecast <- function(y_train, y_test, dates_test) {
  t0   <- now()
  fit  <- Arima(y_train, order = c(1, 0, 1), include.mean = FALSE, method = "ML")
  pred <- as.numeric(forecast(fit, h = length(y_test))$mean)
  run_time <- now() - t0
  
  report_metrics("ARIMA Model (1, 0, 1)", forecast_metrics(y_test, pred), run_time)
  plot_forecast(dates_test, y_test, pred,
                "ARIMA Train-Test Cross Validation", "ARIMA Predicted")
}

SARIMA_Forecast <- function(y_train, y_test, dates_test) {
  t0   <- now()
  fit  <- 
    Arima(
      y_train,
      order = c(1, 0, 2),
      seasonal = list(order = c(1, 1, 0), period = 52),
      method = "ML"
    )
  pred <- as.numeric(forecast(fit, h = length(y_test))$mean)
  run_time <- now() - t0
  
  report_metrics("SARIMA(1,0,2)(1,1,0)[52]", forecast_metrics(y_test, pred), run_time)
  plot_forecast(dates_test, y_test, pred,
                "SARIMA Train-Test Cross Validation", "SARIMA Predicted")
}


length(sarimax_train)                    
anyNA(x_train); anyNA(sarimax_train)
qr(x_train)$rank                         
round(cor(x_train), 2)                   
apply(x_train, 2, sd)                    


SARIMAX_Forecast <- function(y_train, x_train, y_test, x_test, dates_test) {
  stopifnot(length(y_test) == nrow(x_test))
  t0   <- now()
  fit  <- Arima(y_train, xreg = x_train, order = c(1, 0, 2),
                seasonal = list(order = c(1, 1, 0), period = 52), method = "ML")
  pred <- as.numeric(forecast(fit, xreg = x_test, h = nrow(x_test))$mean)
  run_time <- now() - t0
  
  report_metrics("SARIMAX(1,0,2)(1,1,0)[52]", forecast_metrics(y_test, pred), run_time)
  plot_forecast(dates_test, y_test, pred,
                "SARIMAX Train-Test Cross Validation", "SARIMAX Predicted")
}


SARIMA_Walk_Forward <- function(y_train, y_test, dates_test) {
  t0  <- now()
  fit <- Arima(y_train, order = c(1, 0, 2),
               seasonal = list(order = c(1, 1, 0), period = 52), method = "ML")
  
  full  <- ts(c(as.numeric(y_train), y_test), frequency = 52)
  refit <- Arima(full, model = fit)                       # same parameters, no re-estimation
  pred  <- tail(as.numeric(fitted(refit)), length(y_test))
  run_time <- now() - t0
  
  report_metrics("SARIMA Walk Forward", forecast_metrics(y_test, pred), run_time)
  plot_forecast(dates_test, y_test, pred,
                "SARIMA Walk Forward Prediction", "SARIMA Walk-Forward Predicted")
}


ARIMA_Walk_Forward <- function(y_train, y_test, dates_test) {
  t0 <- now()
  

  fit <- tryCatch(
    Arima(y_train, order = c(1, 0, 1), include.mean = FALSE, method = "ML"),
    error = function(e) {
      message("ML failed (", conditionMessage(e), "); retrying with CSS-ML")
      Arima(y_train, order = c(1, 0, 1), include.mean = FALSE, method = "CSS-ML")
    })
  

  full  <- ts(c(as.numeric(y_train), y_test), frequency = frequency(y_train))
  refit <- Arima(full, model = fit)
  pred  <- tail(as.numeric(fitted(refit)), length(y_test))
  run_time <- now() - t0
  
  report_metrics("ARIMA Walk Forward (1, 0, 1)", forecast_metrics(y_test, pred), run_time)
  plot_forecast(dates_test, y_test, pred,
                "ARIMA Walk Forward Prediction", "ARIMA Walk-Forward Predicted")
}


SARIMAX_Walk_Forward <- function(y_train, x_train, y_test, x_test, dates_test) {
  stopifnot(length(y_test) == nrow(x_test))
  t0 <- now()
  
  
  xp <- prep_xreg(x_train, x_test)
  mu <- mean(y_train); s <- sd(y_train)
  y_sc <- ts((as.numeric(y_train) - mu) / s, frequency = 52)
  

  fit <- fit_sarimax(y_sc, xp$train)
  
 
  full_y <- ts(c(as.numeric(y_sc), (y_test - mu) / s), frequency = 52)
  full_x <- rbind(xp$train, xp$test)
  refit  <- Arima(full_y, model = fit, xreg = full_x)
  
  pred <- tail(as.numeric(fitted(refit)), length(y_test)) * s + mu   # un-scale
  run_time <- now() - t0
  
  report_metrics("SARIMAX Walk Forward", forecast_metrics(y_test, pred), run_time)
  plot_forecast(dates_test, y_test, pred,
                "SARIMAX Walk Forward Prediction", "SARIMAX Walk-Forward Predicted")
}



RF_Walk_Forward <- function(data       = clean_data,
                            rf_train_end  = as.Date("2018-12-31"),
                            rf_test_start = as.Date("2019-01-06"),
                            exo_lag    = 0,
                            n_trees    = 150,
                            max_depth  = 5,
                            seed       = 42) {
  t0 <- now()
  
  rf_data <- data %>%
    arrange(week) %>%
    mutate(across(all_of(exo_vars),
                  ~ if (exo_lag > 0) dplyr::lag(.x, exo_lag) else .x),
           y_lag52      = dplyr::lag(heat_index, 52),   # heat_index_weekly.shift(52)
           heat_D       = heat_index - y_lag52,         # seasonal difference
           heat_D_lag52 = dplyr::lag(heat_D, 52)) %>%   # heat_D.shift(52)
    filter(if_all(all_of(c(exo_vars, "y_lag52", "heat_D", "heat_D_lag52")),
                  ~ !is.na(.x)))                        # .dropna()
  
  feat_cols <- c(exo_vars, "heat_D_lag52")
  X <- as.data.frame(rf_data[, feat_cols])
  y <- rf_data$heat_D
  
  n          <- nrow(rf_data)
  n_train    <- sum(rf_data$week <= rf_train_end)
  first_test <- which(rf_data$week >= rf_test_start)[1]
  if (is.na(first_test) || n_train < 10 || first_test != n_train + 1) {
    stop("Train/test split is not contiguous or the training set is too small. ",
         "Check rf_train_end / rf_test_start.")
  }
  
  test_idx <- (n_train + 1):n
  pred     <- numeric(length(test_idx))
  
  for (i in seq_along(test_idx)) {
    t <- test_idx[i]
    train_df <- X[seq_len(t - 1), , drop = FALSE]     # x.iloc[:t]
    train_df$heat_D <- y[seq_len(t - 1)]              # y.iloc[:t]
    
    rf <- ranger(dependent.variable.name = "heat_D", data = train_df,
                 num.trees = n_trees, max.depth = max_depth,
                 mtry = length(feat_cols), min.node.size = 1, seed = seed)
    
    delta   <- predict(rf, data = X[t, , drop = FALSE])$predictions
    pred[i] <- rf_data$y_lag52[t] + delta             # y[t-52] + predicted change
  }
  
  actual   <- rf_data$heat_index[test_idx]
  dates_rf <- rf_data$week[test_idx]
  run_time <- now() - t0
  
  report_metrics("Random Forest Walk Forward", forecast_metrics(actual, pred), run_time)
  plot_forecast(dates_rf, actual, pred,
                "Random Forest Walk Forward Prediction", "Random Forest Forecast")
  invisible(data.frame(week = dates_rf, actual = actual, pred = pred))
}



RF_Walk_Forward <- function(data          = clean_data,
                            rf_train_end  = as.Date("2018-12-31"),
                            rf_test_start = as.Date("2019-01-06"),
                            exo_lag       = 0,
                            n_trees       = 100,
                            max_depth     = 5,
                            seed          = 42,
                            verbose       = TRUE) {
  t0 <- now()
  seasonal_lag <- 52
  feat_cols    <- c(exo_vars, "heat_D_lag52")
  

  rf_data <- data %>%
    arrange(week) %>%
    mutate(across(all_of(exo_vars),
                  ~ if (exo_lag > 0) dplyr::lag(.x, exo_lag) else .x),
           y_lag52      = dplyr::lag(heat_index, seasonal_lag),
           heat_D       = heat_index - y_lag52,
           heat_D_lag52 = dplyr::lag(heat_D, seasonal_lag)) %>%
    filter(if_all(all_of(c(feat_cols, "y_lag52", "heat_D")), ~ !is.na(.x)))
  
  X <- as.data.frame(rf_data[, feat_cols])
  y <- rf_data$heat_D
  

  n          <- nrow(rf_data)
  n_train    <- sum(rf_data$week <= rf_train_end)
  first_test <- which(rf_data$week >= rf_test_start)[1]
  
  if (is.na(first_test) || n_train < 10 || first_test != n_train + 1) {
    stop("Train/test split is not contiguous or the training set is too small. ",
         "Check rf_train_end and rf_test_start.")
  }
  
  test_idx <- (n_train + 1):n
  pred     <- numeric(length(test_idx))
  
 
  for (i in seq_along(test_idx)) {
    t <- test_idx[i]
    
    train_df        <- X[seq_len(t - 1), , drop = FALSE]
    train_df$heat_D <- y[seq_len(t - 1)]
    
    rf <- ranger(dependent.variable.name = "heat_D",
                 data           = train_df,
                 num.trees      = n_trees,
                 max.depth      = max_depth,
                 mtry           = length(feat_cols),
                 min.node.size  = 1,
                 seed           = seed)
    
    delta   <- predict(rf, data = X[t, , drop = FALSE])$predictions
    pred[i] <- rf_data$y_lag52[t] + delta
    
    if (verbose && (i %% 25 == 0 || i == length(test_idx))) {
      message(sprintf("Random Forest: %d / %d steps", i, length(test_idx)))
    }
  }
  

  actual   <- rf_data$heat_index[test_idx]
  dates_rf <- rf_data$week[test_idx]
  run_time <- now() - t0
  
  label <- sprintf("Random Forest Walk Forward (%d trees, depth %d)", n_trees, max_depth)
  report_metrics(label, forecast_metrics(actual, pred), run_time)
  
  plot_args <- list(dates_rf, actual, pred,
                    "Random Forest Walk Forward Prediction", "Random Forest Forecast")
  if (exists("plot_walk_forward")) do.call(plot_walk_forward, plot_args)
  else                             do.call(plot_forecast,     plot_args)
  
  invisible(data.frame(week = dates_rf, actual = actual, pred = pred))
}


ACF_Plot <- function(x, lag_max = 104, title = "Series") {
  x <- as.numeric(x)
  stopifnot(!anyNA(x))
  Acf(x, lag.max = lag_max, main = paste("ACF -", title), xlab = "Lag (weeks)")
  invisible(NULL)
}

PACF_Plot <- function(x, lag_max = 104, title = "Series") {
  x <- as.numeric(x)
  stopifnot(!anyNA(x))
  Pacf(x, lag.max = lag_max, main = paste("PACF -", title), xlab = "Lag (weeks)")
  invisible(NULL)
}  



ADF(heat_D, heat_D_dates)
SARIMA_Search(heat_train_ts)
SARIMAX_Search(sarimax_train, x_train)
ARIMA_Forecast(heat_train_ts, heat_test, dates_test)
SARIMA_Forecast(heat_train_ts, heat_test, dates_test)
SARIMAX_Forecast(sarimax_train, x_train, heat_test, x_test, dates_test)
SARIMA_Walk_Forward(heat_train_ts, heat_test, dates_test)
ARIMA_Walk_Forward(heat_train_ts, heat_test, dates_test)
SARIMAX_Walk_Forward(sarimax_train, x_train, heat_test, x_test, dates_test)

rf_results <- RF_Walk_Forward()                  

rf_results <- RF_Walk_Forward(rf_train_end = train_end, rf_test_start = test_start)


rf_results <- RF_Walk_Forward(rf_train_end = train_end, rf_test_start = test_start, exo_lag = 1) 


RF_Walk_Forward()


fit_arima <- Arima(heat_train_ts, order = c(1, 0, 1), include.mean = FALSE, method = "CSS-ML")

ACF_Plot(residuals(fit_arima),  title = "ARIMA(1,0,1) residuals")
PACF_Plot(residuals(fit_arima), title = "ARIMA(1,0,1) residuals")

ACF_Plot(residuals(fit),  title = "Model residuals")
PACF_Plot(residuals(fit), title = "Model residuals")