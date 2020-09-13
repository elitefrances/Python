print('start.')
import pandas as pd 
import numpy as np 
import csv
import time
import glob


folder = "C:/Users/fding/Desktop/BI Tasks/Monthly/CPM Trend/Jul 2020"
tactic_file = "/query_result.csv"
etv_file = "/ETV7.csv"
dx_file = "/DX7.csv"
dma_mapping = "/dma mapping.csv"
month = "7/1/2020"

tactic_mapping = pd.read_csv(folder+tactic_file)
etv_data = pd.read_csv(folder+etv_file)
dx_data = pd.read_csv(folder+dx_file, low_memory=False)
dma_data = pd.read_csv(folder+dma_mapping)

print ('dx_data.shape[0]', dx_data.shape[0])
print ('etv_data.shape[0]', etv_data.shape[0])


######### manipulation for dx data only contains display ############
dx_data = dx_data.rename(columns = {'Postal Code':'zip code'})
dx_data['zip code'].fillna(0,inplace = True)
pd.options.mode.chained_assignment = None
dx_data['zip code'] = dx_data['zip code'].astype(str)
unique_dma_zip = dma_data.groupby(['zip code'],as_index = False).nth(0)

unique_dma_zip['zip code'] = unique_dma_zip['zip code'].astype(str)
dx_dma = pd.merge(dx_data, unique_dma_zip, on  = 'zip code', how = 'left')

print (dx_data.shape[0],dx_dma.shape[0])

def spend_calculation(df):
	if df['Impressions'] < 100:
		return df['Spend']
	else:
		return (df['Spend']/df['Impressions']*1000-0.35)*df['Impressions']/1000

dx_dma['adjust_spend'] = dx_dma.apply(spend_calculation, axis = 1)
dx_dma = dx_dma.rename(columns = {'dma name':'DMA'})

dx_display = dx_dma[(dx_dma['Creative Format']=='BANNER')|(dx_dma['Creative Format']=='RICH_MEDIA')][['Month','DMA',
				'Tactic Name','Creative Format','Impressions','Clicks','adjust_spend']]
dx_display = dx_display.rename(columns = {'Tactic Name':'Tactic','adjust_spend':'Spend'})


########## manipulation for ETV data contains display/video/OTT #############
etv_data = etv_data.rename(columns = {'dma':'dma code'})
etv_data['dma code'] = etv_data['dma code'].astype(str)
unique_dma_mapping = dma_data[['dma code','dma name']].drop_duplicates()
unique_dma_mapping['dma code'] = unique_dma_mapping['dma code'].apply(str)

etv_dma = pd.merge(etv_data,unique_dma_mapping, on = 'dma code',how = 'left')

tactic_mapping = tactic_mapping.rename(columns = {'adgroup_name':'adgroup','tactic':'Tactic'})

unique_tactic_mapping = tactic_mapping.groupby(['campaign','adgroup'],as_index = False).nth(0)
etv = etv_dma.merge(unique_tactic_mapping,on = ['campaign','adgroup'],how = 'left')

#unique_tactic_mapping = tactic_mapping.groupby('adgroup',as_index = False).nth(0)
#etv = pd.merge(etv_dma,unique_tactic_mapping[['adgroup','Tactic']], on = 'adgroup', how = 'left')

print (etv_data.shape[0], etv_dma.shape[0], etv.shape[0])


def media_type(df):
	if '_OTT_V2' in df['campaign']:
		return 'OTT'
	elif '_CDV_BLEND' in df['campaign']:
		return 'CDV'
	elif '_Video' in df['campaign']:
		return 'Video'
	else:
		return 'Display'

etv['media_type'] = etv.apply(media_type,axis = 1)
etv_display = etv[etv['media_type']=='Display'][['date_part','dma name','Tactic','media_type','impressions','clicks','spend']]
etv_display = etv_display.rename(columns = {'date_part':'Month','dma name':'DMA',
	'media_type':'Creative Format','impressions':'Impressions','clicks':'Clicks','spend':'Spend'})

etv_video = etv[etv['media_type']=='Video'][['date_part','dma code','impressions','spend','dma name','Tactic']]
etv_video = etv_video.rename(columns = {'date_part':'Month','dma code':'DMA',
	'impressions':'Impressions','spend':'Spend','dma name':'DMA Code'})

etv_ott = etv[etv['media_type']=='OTT'][['date_part','dma code','impressions','spend','dma name','Tactic']]
etv_ott = etv_ott.rename(columns = {'date_part':'Month','dma code':'DMA',
	'impressions':'Impressions','spend':'Spend','dma name':'DMA Code'})

etv_cdv = etv[etv['media_type']=='CDV'][['date_part','dma code','impressions','spend','dma name','Tactic']]
etv_cdv = etv_cdv.rename(columns = {'date_part':'Month','dma code':'DMA',
	'impressions':'Impressions','spend':'Spend','dma name':'DMA Code'})

display = dx_display.append(etv_display)

print (display.shape[0] + etv_video.shape[0] + etv_ott.shape[0] + etv_cdv.shape[0])

display.to_csv(folder +'/' + 'display_raw.csv',index = False)
etv_video.to_csv(folder +'/' + 'video_raw.csv',index = False)
etv_ott.to_csv(folder +'/' + 'ott_raw.csv',index = False)
etv_cdv.to_csv(folder + '/' + 'cdv_raw.csv', index = False)

############### manipulation facebook data ###################

files = glob.glob(folder + '/FB/*.csv')
if len(files) == 0:
	raise Exception('!!!FB files format is wrong!!!')
print (files)
fb1 = pd.read_csv(files[0])
if len(files) > 1:
	fb2 = pd.read_csv(files[1])

print (fb1.shape[0])
if len(files) > 1:
	print (fb2.shape[0])

fb_master = fb1
if len(files) > 1:
	fb_master = fb_master.append(fb2)

def drive_auto(df):
	if df['Objective']=='PRODUCT_CATALOG_SALES':
		return "Drive Auto"
	else:
		return "None Drive Auto"

def do_not_use(df):
	if 'do' in df['Ad Set Name'].lower() and 'not' in df['Ad Set Name'].lower():
		return 'not use'
	else:
		return 0		

def fb_type(df):
	if '_video' in str(df['Campaign Name']).lower():
		return 'Video'
	elif '_display' in str(df['Campaign Name']).lower():
		return 'Display'
	else:
		return ""

fb_master['Drive auto'] = fb_master.apply(drive_auto,axis = 1)
fb_master['Media'] = fb_master.apply(fb_type,axis = 1)
fb_master['Do not use'] = fb_master.apply(do_not_use,axis = 1)

fb_output = fb_master[['Reporting Starts','Reporting Ends','Campaign Name','Ad Set Name','Media',
					'Objective','Drive auto','Do not use','Delivery Status',"CPM (Cost per 1,000 Impressions)",'Impressions',	
					"Amount Spent (USD)", "Clicks (All)","Frequency","Unique Link Clicks","Link Clicks"]]
print (fb_output.shape[0])

fb_output.to_csv(folder+'/FB/fb_raw.csv',index = False)

print ('cpm_process complete.')
