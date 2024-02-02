# Automatically load the renv environment
renv::load()

library("apsimx")
library(data.table)
library(sf)
library(readxl)
library(lubridate)
library(zoo)

base_dir = getwd()
regions_sf = sf::st_read("./mt regions/mt regions.shp")
regions_sf <- sf::st_centroid(regions_sf)

colnames(regions_sf)[colnames(regions_sf) == "id_regiao_"] <- "region_id"


file_path <- "MT Soybeans Dataset.xlsx"
planting_dt <- as.data.table(read_excel(path = file_path, sheet = "planting"))

weather_dt <- as.data.table(read_excel(path = file_path, sheet = "weather"))
weather_dt[, date := as.Date(date)]
weather_dt[, year := year(date)]
weather_dt[, day := yday(date)]
setnames(weather_dt, c("prcp", "tmax", "tmin"), c("rain", "maxt", "mint"))

# Select the earliest date for each year
planting50_dt <- planting_dt[pct_planting_cumulative > 0.4, .SD[which.min(date)], by = .(crop_year,region_id )]
planting50_dt[, year := year(date)]


region_n = 5101
year_n = 2023
cult_n = "MG6"
run_one_region <- function(region_n){
  dir.create(paste0(base_dir,"/apsim_run/sim_rds/"))
  dir.create(paste0(base_dir,"/apsim_run/apsim_files/"))
  
  print(region_n)
  one_loc_sf <- subset(regions_sf, region_id == region_n)
  
  coords <- sf::st_coordinates(one_loc_sf)
  class(coords)
  region_name = tolower(as.character(one_loc_sf$nm_regiao_))
  
  fpath_met = paste0(base_dir,"/apsim_run/met_files" )
  fname_met = paste0(region_name, '.met')
  met_full_path <- file.path(fpath_met, fname_met)
  
  power_met <- apsimx::get_power_apsim_met(lonlat = c(coords[,"X"],coords[,"Y"]), # -5.5498, 33.9131
                              dates = c("2010-01-01", "2024-01-13"), 
                              wrt.dir = fpath_met,
                              filename=fname_met )
  power_met = data.table(power_met)
  power_met = power_met[,c("year", "day", "radn")]
  
  weather_tmp = weather_dt[region_id == region_n]
  # Left join on 'year' and 'day'
  weather_tmp2 <- weather_tmp[power_met, on = .(year, day)]
  #ffill missing radn
  weather_tmp2[, radn := na.locf(radn, na.rm = FALSE)]
  weather_tmp2 = weather_tmp2[,c("year", "day", "radn", "maxt", "mint", "rain")]
  
  user_met = as_apsim_met(
    weather_tmp2,
    filename = met_full_path,
    site = "nosite",
    latitude = coords[,"Y"],
    longitude = coords[,"X"],
    # tav = NA,
    # amp = NA,
    colnames = c("year", "day", "radn", "maxt", "mint", "rain"),
    units = c("()", "()", "(MJ/m2/day)", "(oC)", "(oC)", "(mm)"),
    # constants = NA,
    # comments = NA,
    # check = TRUE
  )
  write_apsim_met(user_met,wrt.dir = fpath_met,filename=fname_met)
  
  ## Check the met file 
  apsimx::check_apsim_met(user_met)
  
  year_ls = as.list(seq(2010, 2023))
  cult_ls = c("MG5", "MG6", "MG7", "MG8", "MG9")
  # Create a list of years from 2010 to 2023
  for( year_n in year_ls){
    for( cult_n in cult_ls){
      sim_id = paste0(region_name,"_", cult_n ,"_",year_n)
      path_rds = paste0(base_dir,"/apsim_run/sim_rds/",sim_id, ".rds" )
      if (file.exists(path_rds)) {
        next  # Move to the next iteration if the file does not exist
      }
      
      plant_date = planting50_dt[ region_id == region_n & year==year_n, date]
      
      if (length(plant_date) == 0) {
        # Extract day of year for each date in the specified region
        doy <- as.numeric(format(planting50_dt[region_id == region_n, date], "%j"))
        doy_mean <- mean(doy)
        # If plant_date not found for the specified year, calculate mean date
        plant_date <- as.Date(paste(year_n, "-01-01", sep = "") , format = "%Y-%m-%d") + doy_mean - 1
  
      }
        
      # New folder
      fpath_apsim = paste0(base_dir,"/apsim_run/apsim_files/", sim_id )
      # Delete folder
      unlink(fpath_apsim, recursive = TRUE, force=TRUE)
      
      dir.create(fpath_apsim)
      fname_apsim = paste0(sim_id, '.apsimx')
      apsim_full_path <- file.path(fpath_apsim, fname_apsim)
      file.copy("./apsim_run/brazil_base_v2.apsimx", apsim_full_path) 
      
      # Update Met file
      apsimx::inspect_apsimx(file = fname_apsim, 
                              src.dir = fpath_apsim, 
                              node = "Weather")
      
      
      apsimx::edit_apsimx(file = fname_apsim, 
                          src.dir = fpath_apsim, 
                  node = "Weather",
                  value = met_full_path,
                  overwrite = TRUE,
                  )
      
      # Update Clock
      apsimx::inspect_apsimx(file = fname_apsim,  src.dir = fpath_apsim , node = "Clock")
      
      new_dates = c(paste0(year_n, "-03-01T00:00:00"),paste0(year_n+1, "-05-28T00:00:00"))
      
      apsimx::edit_apsimx(file = fname_apsim, 
                  src.dir = fpath_apsim , 
                  node = "Clock",
                  parm = c("Start", "End"),
                  value = new_dates,
                  overwrite = TRUE)
      
      # Update Planting Date
      ## Change the sowing rule for when rain is available
      # Convert to Date object
      date_obj <- as.Date(plant_date, format = "%Y-%m-%d")
      formatted_date <- format(date_obj, "%e-%b")
      
      apsimx::edit_apsimx(file = fname_apsim, src.dir = fpath_apsim ,
                  node = "Manager",
                  manager.child = "Sow on a fixed date",
                  parm = "SowDate", ## This is for start date
                  value = formatted_date,
                  overwrite = TRUE)
      
      # Edit CUltivar
      cultivar_str = paste0("Generic_", cult_n)
      apsimx::edit_apsimx(file = fname_apsim, src.dir = fpath_apsim ,
                          node = "Manager",
                          manager.child = "Sow on a fixed date",
                          parm = "CultivarName", ## This is for start date
                          value = cultivar_str,
                          overwrite = TRUE)
      
      apsimx::inspect_apsimx(file = fname_apsim, src.dir = fpath_apsim , node = "Manager",  parm = list("Sow on a fixed date", NA))
      
      # Update initial water
      # apsimx::inspect_apsimx(file = fname_apsim, src.dir = fpath_apsim , node = "Soil", soil.child="Water")
      
      out_dt <- apsimx::apsimx(file = fname_apsim, 
                    src.dir = fpath_apsim, )
      
      out_dt = as.data.table(out_dt)
      out_dt[, year := year_n]
      out_dt[, cultivar := cult_n]
      out_dt[, region_id := region_n]
      out_dt[, region_name := region_name]
      # sim_ls[[as.character(year_n)]] <- out_dt
      
      saveRDS(out_dt, file = path_rds)
    } # end cult_n
  } # end year_n
}

# Initialize an empty list to store results

region_ls = unique(planting50_dt$region_id)
# Apply the function to each combination and store the results
for (region_n in region_ls) {
  run_one_region(region_n)
}

load_sim_results <- function(folder_path){
  # List all RDS files in the directory
  rds_files <- list.files(folder_path, pattern = "\\.rds$", full.names = TRUE)
  
  # Load each RDS file and store in a list
  result_list <- lapply(rds_files, readRDS)
  
  # Combine the list of results into a single data.table
  sim_dt <- rbindlist(result_list)
  return (sim_dt)
}

folder_path = paste0(base_dir,"/apsim_run/sim_rds/")
sim_dt = load_sim_results(folder_path)
sim_dt[,.N, by=.(region_id)]
path_rds = paste0(base_dir,"/apsim_run/sim_dul_weather_MG6.rds")
saveRDS(sim_dt, file = path_rds)
unlink(paste0(base_dir,"/apsim_run/sim_rds/"), recursive = TRUE, force=TRUE)
unlink(paste0(base_dir,"/apsim_run/apsim_files/"), recursive = TRUE, force=TRUE)
