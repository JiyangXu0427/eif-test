# Title     : TODO
# Objective : TODO
# Created by: ThinkPad
# Created on: 5/6/2021


## --------------------
library(sROC);
library(R.matlab);

#filename <- "annthyroid"
#filename <- "cardio"
#filename <- "covtype"
filename <- "ionosphere"
#filename <- "mnist"
#filename <- "satellite"
#filename <- "shuttle"
#filename <- "thyroid"

dataset_folder_path <- "./datasets/"
file_suffix <- ".mat"
file_path <- paste(dataset_folder_path,filename,file_suffix,sep="")
#file_path <- './datasets/thyroid.mat';
raw_data <- readMat(file_path);
x_values <- raw_data$X;
skip_value <- 0
counter <- 1;

cdf_y_results_list <- list()
cdf_x_results_list <- list()
while(counter <= ncol(x_values)) {
  info <- paste(counter,"start",sep="")
  print(info)

  column_value <- x_values[,counter]

  if(filename == "cardio"){
    if (counter == 6 || counter == 16 || counter == 21 ){
      counter <- counter + 1
      skip_value <- skip_value + 1
      print(skip_value)
      print("next_reached")
      next
    }
  }
  if(filename == "ionosphere"){
    if (counter == 1 ){
      counter <- counter + 1
      skip_value <- skip_value + 1
      next
    }
  }

  if(filename == "mnist"){
    if (counter == 1 || counter == 4 || counter == 7
      || counter == 8 || counter == 16 || counter == 22
      || counter == 27 || counter == 29 || counter == 30
      || counter == 32 || counter == 36 || counter == 37
      || counter == 40 || counter == 41 || counter == 45
      || counter == 51 || counter == 53 || counter == 54
      || counter == 56 || counter == 61 || counter == 62
      || counter == 63 || counter == 71 || counter == 73
      || counter == 79 || counter == 83 || counter == 84
      || counter == 87 || counter == 88 || counter == 89
      || counter == 90 || counter == 92 || counter == 98
      || counter == 100
    ){
      counter <- counter + 1
      skip_value <- skip_value + 1
      next
    }
  }

  numbers_of_CDF_points <- length(column_value) * 10
  #numbers_of_CDF_points <- 4
  x.CDF <- kCDF(column_value,ngrid = numbers_of_CDF_points)
  cdf_fhat <- x.CDF$Fhat
  cdf_y_results_list <- c(cdf_y_results_list, cdf_fhat)
  cdf_x <- x.CDF$x
  cdf_x_results_list <- c(cdf_x_results_list, cdf_x)
  info <- paste(counter,"done",sep="")
  print(info)
  counter <- counter + 1
}

result_folder_path <- "./plots/"

csv_filename_y <- paste(filename, "-CDF_points_y.csv", sep="")
csv_file_path_y <- paste(result_folder_path, csv_filename_y, sep="")

csv_filename_x <- paste(filename, "-CDF_points_x.csv", sep="")
csv_file_path_x <- paste(result_folder_path, csv_filename_x, sep="")
print("begeing generating matrix")

ncol_value <- ncol(x_values)  - skip_value

y_maxtrix_result <- matrix(unlist(cdf_y_results_list), ncol = ncol_value )
x_maxtrix_result <- matrix(unlist(cdf_x_results_list), ncol = ncol_value )
print("begeing generating csv")
write.csv(y_maxtrix_result, csv_file_path_y)
write.csv(x_maxtrix_result, csv_file_path_x)
print("generating csv done")