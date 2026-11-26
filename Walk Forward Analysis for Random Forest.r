
install.packages(c("readr", "dplyr", "ggplot2", "forecast", "ranger"))
library(readr)     # read_csv                     (pd.read_csv)
library(dplyr)     # wrangling, lags, resample    (pandas)
library(ranger)   # random forest                (sklearn RandomForestRegressor)
library(ggplot2)   # plots                        (matplotlib)

csv_path <- #File Directory Here

data <- read_csv(csv_path,
                 col_types = cols(date = col_character(), .default = col_guess())) %>%
  mutate(date = as.Date(date, format = "%d/%m/%Y")) %>%
  select(date, heat_index, solar_rad, wind_speed, uv_rad)

vars     <- c("heat_index", "solar_rad", "wind_speed", "uv_rad")
exo_vars <- c("solar_rad", "wind_speed", "uv_rad")

clean_data <- data %>%
  mutate(week = date + (7 - as.integer(format(date, "%u")))) %>%
  group_by(week) %>%
  summarise(across(all_of(vars), ~ mean(.x, na.rm = TRUE)), .groups = "drop") %>%
  arrange(week) %>%
  filter(if_all(all_of(vars), ~ !is.na(.x)))
new_data <- clean_data %>%
  mutate(D          = heat_index - dplyr::lag(heat_index, 52),
         heat_lag52 = dplyr::lag(heat_index, 52)) %>%
  transmute(
    week, heat_index, heat_lag52, D,
    across(all_of(exo_vars), ~ dplyr::lag(.x, 52), .names = "{.col}_lag52"),
    across(all_of(exo_vars), ~ dplyr::lag(.x, 1),  .names = "{.col}_lag1"),
    D_lag1  = dplyr::lag(D, 1),
    D_lag52 = dplyr::lag(D, 52)
  ) %>%
  filter(if_all(-week, ~ !is.na(.x))) %>%
  as.data.frame()

feature_cols <- c(paste0(exo_vars, "_lag52"),
                  paste0(exo_vars, "_lag1"),
                  "D_lag1", "D_lag52")
train_end <- as.Date("2019-12-29")      
n_train   <- sum(new_data$week <= train_end)
n_total   <- nrow(new_data)
stopifnot(n_train > 0, n_train < n_total)

test_idx   <- (n_train + 1):n_total
test_dates <- new_data$week[test_idx]
actual     <- new_data$heat_index[test_idx]
start <- proc.time()[["elapsed"]]

forecast_rf <- numeric(length(test_idx))

for (i in test_idx) {
  train_df <- new_data[seq_len(i - 1), c("D", feature_cols)]
  
  model <- ranger(
    D ~ .,
    data          = train_df,
    num.trees     = 150,
    max.depth     = 5,
    mtry          = length(feature_cols),
    min.node.size = 1,
    replace       = TRUE,
    seed          = 42,
    verbose       = FALSE
  )
  
  delta_prediction <- predict(model, data = new_data[i, feature_cols, drop = FALSE])$predictions
  forecast_rf[i - n_train] <- new_data$heat_lag52[i] + delta_prediction
}

run_time <- proc.time()[["elapsed"]] - start
df <- data.frame(date = test_dates, actual = actual, pred = forecast_rf)

p <- ggplot(df, aes(x = date)) +
  geom_line(aes(y = actual, colour = "Actual (True)"), linewidth = 0.8) +
  geom_line(aes(y = pred, colour = "Random Forest Forecast"),
            linewidth = 1, linetype = "dashed") +
  scale_colour_manual(values = c("Actual (True)" = "black",
                                 "Random Forest Forecast" = "#d9534f"),
                      name = NULL) +
  labs(title = "Random Forest Walk Forward Prediction",
       x = "Date", y = "Heat Index (\u00B0C)") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
print(p)

err       <- forecast_rf - actual
rmse      <- sqrt(mean(err^2))
mae       <- mean(abs(err))
bias      <- mean(err)
maxerror  <- max(abs(err))
r_squared <- 1 - sum(err^2) / sum((actual - mean(actual))^2)

cat(sprintf("Random Forest Regressor RMSE: %s\n",       format(rmse,      digits = 7)))
cat(sprintf("Random Forest Regressor MAE: %s\n",        format(mae,       digits = 7)))
cat(sprintf("Random Forest Regressor Bias: %s\n",       format(bias,      digits = 7)))
cat(sprintf("Random Forest Regressor Max Error: %s\n",  format(maxerror,  digits = 7)))
cat(sprintf("Random Forest Regressor R-Squared: %s\n",  format(r_squared, digits = 7)))
cat(sprintf("Random Forest Regressor Run Time: %.3fs\n", run_time))