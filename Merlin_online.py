import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import librosa
import os
# import csv
import datetime
import time
import math
import requests

# def initialize_session_state():
    # st.session_state.date = datetime.datetime.now()
    # st.session_state.lat = 37.299
    # st.session_state.lon = -75.929
    # st.session_state.min_conf = 30
    # st.session_state.sf_thresh = 3
    # st.session_state.res_type = "soxr_hq"

classifications = ["bird sp.","Great Tinamou","Little Tinamou","Brown Tinamou","Undulated Tinamou","Thicket Tinamou","Tataupa Tinamou","Black-bellied Whistling-Duck","Fulvous Whistling-Duck","Snow Goose","Graylag Goose","Greater White-fronted Goose","Pink-footed Goose","Brant","Barnacle Goose","Cackling Goose","Canada Goose","Mute Swan","Trumpeter Swan","Tundra Swan","Whooper Swan","Egyptian Goose","Wood Duck","Northern Shoveler","Gadwall","Eurasian Wigeon","American Wigeon","Mallard","Northern Pintail","Green-winged Teal","Common Eider","Black Scoter","Long-tailed Duck","Ruddy Duck","Plain Chachalaca","Chaco Chachalaca","Speckled Chachalaca","Spix's Guan","Dusky-legged Guan","Mountain Quail","Northern Bobwhite","California Quail","Gambel's Quail","Montezuma Quail","Wild Turkey","Ruffed Grouse","Gray Partridge","Ring-necked Pheasant (Ring-necked)","Indian Peafowl","Red Junglefowl","Gray Junglefowl","Gray Francolin","Black Francolin","Painted Francolin","Common Quail","Red-legged Partridge","Chukar","Little Grebe","Least Grebe","Pied-billed Grebe","Red-necked Grebe","Great Crested Grebe","Western Grebe","Rock Pigeon","Stock Dove","Common Wood-Pigeon","Pale-vented Pigeon","Scaled Pigeon","Red-billed Pigeon","Band-tailed Pigeon","Plumbeous Pigeon","Short-billed Pigeon","European Turtle-Dove","Oriental Turtle-Dove","Eurasian Collared-Dove","Spotted Dove","Laughing Dove","Asian Emerald Dove","Inca Dove","Common Ground Dove","Ruddy Ground Dove","White-tipped Dove","Gray-fronted Dove","White-winged Dove","Mourning Dove","Guira Cuckoo","Smooth-billed Ani","Groove-billed Ani","Striped Cuckoo","Pheasant Cuckoo","Pavonine Cuckoo","Lesser Ground-Cuckoo","Greater Roadrunner","Greater Coucal","Pied Cuckoo","Squirrel Cuckoo","Yellow-billed Cuckoo","Mangrove Cuckoo","Black-billed Cuckoo","Asian Koel","Gray-bellied Cuckoo","Large Hawk-Cuckoo","Common Hawk-Cuckoo","Indian Cuckoo","Common Cuckoo","Lesser Nighthawk","Common Nighthawk","Antillean Nighthawk","Common Pauraque","Common Poorwill","Chuck-will's-widow","Rufous Nightjar","Buff-collared Nightjar","Eastern Whip-poor-will","Mexican Whip-poor-will","Red-necked Nightjar","Eurasian Nightjar","Great Potoo","Common Potoo","White-collared Swift","Chimney Swift","Vaux's Swift","Alpine Swift","Common Swift","Pallid Swift","White-throated Swift","Asian Palm-Swift","Sparkling Violetear","Mexican Violetear","Lesser Violetear","Tyrian Metaltail","Blue-throated Mountain-gem","Ruby-throated Hummingbird","Black-chinned Hummingbird","Anna's Hummingbird","Costa's Hummingbird","Calliope Hummingbird","Rufous Hummingbird","Allen's Hummingbird","Broad-tailed Hummingbird","Broad-billed Hummingbird","Ridgway's Rail","King Rail","Clapper Rail","Virginia Rail","Water Rail","Corn Crake","Plumbeous Rail","Giant Wood-Rail","Gray-cowled Wood-Rail","Sora","Spotted Crake","Eurasian Moorhen","Common Gallinule","Eurasian Coot","American Coot","Purple Gallinule","White-breasted Waterhen","Yellow Rail","Rufous-sided Crake","Black Rail","Limpkin","Sandhill Crane","Common Crane","Eurasian Thick-knee","Black-winged Stilt","Black-necked Stilt","Pied Avocet","American Avocet","Eurasian Oystercatcher","American Oystercatcher","Black Oystercatcher","Black-bellied Plover","European Golden-Plover","American Golden-Plover","Northern Lapwing","Yellow-wattled Lapwing","Red-wattled Lapwing","Southern Lapwing","Common Ringed Plover","Semipalmated Plover","Piping Plover","Little Ringed Plover","Killdeer","Upland Sandpiper","Whimbrel","Long-billed Curlew","Eurasian Curlew","Black-tailed Godwit","Marbled Godwit","Ruddy Turnstone","Black Turnstone","Sanderling","Dunlin","Least Sandpiper","Semipalmated Sandpiper","Western Sandpiper","Short-billed Dowitcher","Long-billed Dowitcher","Eurasian Woodcock","American Woodcock","Common Snipe","Wilson's Snipe","Common Sandpiper","Spotted Sandpiper","Green Sandpiper","Solitary Sandpiper","Spotted Redshank","Greater Yellowlegs","Common Greenshank","Willet","Lesser Yellowlegs","Wood Sandpiper","Common Redshank","Marbled Murrelet","Black-legged Kittiwake","Bonaparte's Gull","Black-headed Gull","Laughing Gull","Franklin's Gull","Common Gull","Ring-billed Gull","Western Gull","California Gull","Herring Gull","Yellow-legged Gull","Lesser Black-backed Gull","Glaucous-winged Gull","Great Black-backed Gull","Little Tern","Least Tern","Gull-billed Tern","Caspian Tern","Black Tern","Common Tern","Arctic Tern","Forster's Tern","Royal Tern","Sandwich Tern","Black Skimmer","Red-throated Loon","Common Loon","Anhinga","American Bittern","Great Bittern","Little Bittern","Least Bittern","Great Blue Heron","Gray Heron","Purple Heron","Great Egret","Little Egret","Snowy Egret","Little Blue Heron","Tricolored Heron","Green Heron","Black-crowned Night-Heron","Yellow-crowned Night-Heron","White-faced Ibis","Osprey","White-tailed Kite","Oriental Honey-buzzard","Crested Serpent-Eagle","Changeable Hawk-Eagle","Black Hawk-Eagle","Ornate Hawk-Eagle","Booted Eagle","Snail Kite","Mississippi Kite","Eurasian Marsh-Harrier","Northern Harrier","Shikra","Eurasian Sparrowhawk","Sharp-shinned Hawk","Cooper's Hawk","Northern Goshawk","Black Kite","Bald Eagle","Roadside Hawk","Harris's Hawk","Gray Hawk","Gray-lined Hawk","Red-shouldered Hawk","Broad-winged Hawk","Swainson's Hawk","Zone-tailed Hawk","Red-tailed Hawk","Common Buzzard","Barn Owl","Indian Scops-Owl","Eurasian Scops-Owl","Flammulated Owl","Whiskered Screech-Owl","Tropical Screech-Owl","Western Screech-Owl","Eastern Screech-Owl","Spectacled Owl","Great Horned Owl","Eurasian Eagle-Owl","Eurasian Pygmy-Owl","Northern Pygmy-Owl","Ferruginous Pygmy-Owl","Jungle Owlet","Elf Owl","Spotted Owlet","Little Owl","Burrowing Owl","Mottled Owl","Black-and-white Owl","Tawny Owl","Spotted Owl","Barred Owl","Long-eared Owl","Short-eared Owl","Boreal Owl","Northern Saw-whet Owl","Morepork","Eared Quetzal","Black-headed Trogon","Gartered Trogon","Black-throated Trogon","Elegant Trogon","Collared Trogon","Eurasian Hoopoe","Indian Gray Hornbill","Lesson's Motmot","Rufous Motmot","Common Kingfisher","White-throated Kingfisher","Ringed Kingfisher","Belted Kingfisher","Green Kingfisher","European Bee-eater","Black-fronted Nunbird","Rufous-tailed Jacamar","Coppersmith Barbet","Great Barbet","Brown-headed Barbet","White-cheeked Barbet","Blue-throated Barbet","Gilded Barbet","Collared Aracari","Toco Toucan","Yellow-throated Toucan","White-throated Toucan","Keel-billed Toucan","Channel-billed Toucan","Eurasian Wryneck","Williamson's Sapsucker","Yellow-bellied Sapsucker","Red-naped Sapsucker","Red-breasted Sapsucker","Lewis's Woodpecker","Red-headed Woodpecker","Acorn Woodpecker","Yellow-tufted Woodpecker","Red-crowned Woodpecker","Gila Woodpecker","Golden-fronted Woodpecker","Red-bellied Woodpecker","American Three-toed Woodpecker","Black-backed Woodpecker","Middle Spotted Woodpecker","White-backed Woodpecker","Great Spotted Woodpecker","Syrian Woodpecker","Lesser Spotted Woodpecker","Downy Woodpecker","Nuttall's Woodpecker","Ladder-backed Woodpecker","Red-cockaded Woodpecker","Hairy Woodpecker","White-headed Woodpecker","Smoky-brown Woodpecker","Greater Flameback","Crimson-crested Woodpecker","Black-rumped Flameback","Gray-headed Woodpecker","Eurasian Green Woodpecker","Iberian Green Woodpecker","Lineated Woodpecker","Pileated Woodpecker","Black Woodpecker","Golden-olive Woodpecker","Green-barred Woodpecker","Northern Flicker","Gilded Flicker","Red-legged Seriema","Laughing Falcon","Collared Forest-Falcon","Barred Forest-Falcon","Yellow-headed Caracara","Eurasian Kestrel","American Kestrel","Merlin","Peregrine Falcon","Alexandrine Parakeet","Rose-ringed Parakeet","Malabar Parakeet","Vernal Hanging-Parrot","Monk Parakeet","Plain Parakeet","Yellow-chevroned Parakeet","Orange-chinned Parakeet","Blue-headed Parrot","Red-crowned Parrot","Red-lored Parrot","Turquoise-fronted Parrot","Mealy Parrot","Orange-winged Parrot","Orange-fronted Parakeet","Blue-and-yellow Macaw","Chestnut-fronted Macaw","Scarlet Macaw","Blue-crowned Parakeet","Green Parakeet","Mitred Parakeet","Red-masked Parakeet","White-eyed Parakeet","Indian Pitta","Fasciated Antshrike","Spot-backed Antshrike","Tufted Antshrike","Great Antshrike","Barred Antshrike","Rufous-capped Antshrike","Variable Antshrike","Rusty-winged Antwren","Rufous-margined Antwren","Dusky Antbird","Gray Antbird","Black-faced Antbird","Chestnut-backed Antbird","Olive-crowned Crescentchest","Mayan Antthrush","Black-faced Antthrush","Short-tailed Antthrush","Olivaceous Woodcreeper","Wedge-billed Woodcreeper","Strong-billed Woodcreeper","Cocoa Woodcreeper","Buff-throated Woodcreeper","Ivory-billed Woodcreeper","Narrow-billed Woodcreeper","Plain Xenops","Pale-legged Hornero","Rufous Hornero","Wren-like Rushbird","Lineated Foliage-gleaner","Buff-throated Foliage-gleaner","White-eyed Foliage-gleaner","Thorn-tailed Rayadito","Greater Thornbird","Stripe-crowned Spinetail","Slaty Spinetail","Rufous-capped Spinetail","Spix's Spinetail","Pale-breasted Spinetail","Sooty-fronted Spinetail","Azara's Spinetail","Rufous-breasted Spinetail","Long-tailed Manakin","Swallow-tailed Manakin","Andean Cock-of-the-rock","Purple-throated Fruitcrow","Screaming Piha","Masked Tityra","Northern Schiffornis","White-winged Becard","Rose-throated Becard","Royal Flycatcher","Slaty-capped Flycatcher","Scale-crested Pygmy-Tyrant","Northern Bentbill","Pearly-vented Tody-Tyrant","Ochre-faced Tody-Flycatcher","Common Tody-Flycatcher","Northern Beardless-Tyrannulet","Southern Beardless-Tyrannulet","Tufted Tit-Tyrant","Yellow-crowned Tyrannulet","Forest Elaenia","Greenish Elaenia","Yellow-bellied Elaenia","Small-billed Elaenia","Large Elaenia","White-crested Elaenia","Mountain Elaenia","Straneck's Tyrannulet","Rufous-crowned Pygmy-Tyrant","Euler's Flycatcher","Tufted Flycatcher","Olive-sided Flycatcher","Greater Pewee","Smoke-colored Pewee","Western Wood-Pewee","Tropical Pewee (Short-legged)","Eastern Wood-Pewee","Yellow-bellied Flycatcher","Acadian Flycatcher","Alder Flycatcher","Willow Flycatcher","Least Flycatcher","Hammond's Flycatcher","Gray Flycatcher","Dusky Flycatcher","Pacific-slope Flycatcher","Cordilleran Flycatcher","Buff-breasted Flycatcher","Black Phoebe","Eastern Phoebe","Say's Phoebe","Vermilion Flycatcher","Bright-rumped Attila","Dusky-capped Flycatcher","Ash-throated Flycatcher","Great Crested Flycatcher","Brown-crested Flycatcher","Lesser Kiskadee","Great Kiskadee","Boat-billed Flycatcher","Rusty-margined Flycatcher","Social Flycatcher","Streaked Flycatcher","Sulphur-bellied Flycatcher","Piratic Flycatcher","Tropical Kingbird","Couch's Kingbird","Cassin's Kingbird","Thick-billed Kingbird","Western Kingbird","Eastern Kingbird","Gray Kingbird","Scissor-tailed Flycatcher","Tui","Small Minivet","Large Cuckooshrike","Black-headed Cuckooshrike","Rufous-browed Peppershrike","Green Shrike-Vireo","Lesser Greenlet","Golden Vireo","Black-capped Vireo","White-eyed Vireo","Mangrove Vireo","Bell's Vireo","Gray Vireo","Hutton's Vireo","Yellow-throated Vireo","Cassin's Vireo","Blue-headed Vireo","Plumbeous Vireo","Philadelphia Vireo","Warbling Vireo","Red-eyed Vireo","Chivi Vireo","Yellow-green Vireo","Black-whiskered Vireo","Eurasian Golden Oriole","Indian Golden Oriole","Black-hooded Oriole","Ashy Woodswallow","White-throated Fantail","Spot-breasted Fantail","Black-naped Monarch","Red-backed Shrike","Loggerhead Shrike","Northern Shrike","Iberian Gray Shrike","Great Gray Shrike","Canada Jay","White-throated Magpie-Jay","Brown Jay","Green Jay","Yucatan Jay","Purplish Jay","Violaceous Jay","Plush-crested Jay","Pinyon Jay","Steller's Jay","Blue Jay","Florida Scrub-Jay","California Scrub-Jay","Woodhouse's Scrub-Jay","Mexican Jay","Eurasian Jay","Iberian Magpie","Rufous Treepie","Gray Treepie","Eurasian Magpie","Black-billed Magpie","Yellow-billed Magpie","Clark's Nutcracker","Eurasian Nutcracker","Red-billed Chough","Yellow-billed Chough","Eurasian Jackdaw","Rook","American Crow","Fish Crow","Chihuahuan Raven","Carrion Crow","Hooded Crow","Common Raven","Gray-headed Canary-Flycatcher","Coal Tit","Crested Tit","Marsh Tit","Willow Tit","Carolina Chickadee","Black-capped Chickadee","Mountain Chickadee","Mexican Chickadee","Chestnut-backed Chickadee","Boreal Chickadee","Eurasian Blue Tit","Bridled Titmouse","Oak Titmouse","Juniper Titmouse","Tufted Titmouse","Black-crested Titmouse","Green-backed Tit","Great Tit","Cinereous Tit","Verdin","Eurasian Penduline-Tit","Jerdon's Bushlark","Indian Bushlark","Horned Lark","Greater Short-toed Lark","Calandra Lark","Wood Lark","Eurasian Skylark","Thekla's Lark","Bearded Reedling","Common Tailorbird","Gray-breasted Prinia","Jungle Prinia","Ashy Prinia","Zitting Cisticola","Eastern Olivaceous Warbler","Melodious Warbler","Icterine Warbler","Moustached Warbler","Sedge Warbler","Marsh Warbler","Eurasian Reed Warbler","Great Reed Warbler","Savi's Warbler","Common Grasshopper-Warbler","Blue-and-white Swallow","Northern Rough-winged Swallow","Purple Martin","Gray-breasted Martin","Tree Swallow","Violet-green Swallow","Bank Swallow","Barn Swallow","Red-rumped Swallow","Cliff Swallow","Cave Swallow","Common House-Martin","Red-vented Bulbul","Red-whiskered Bulbul","White-browed Bulbul","Wood Warbler","Western Bonelli's Warbler","Yellow-browed Warbler","Hume's Warbler","Willow Warbler","Canary Islands Chiffchaff","Common Chiffchaff","Iberian Chiffchaff","Whistler's Warbler","Green Warbler","Greenish Warbler","Arctic Warbler","Gray-hooded Warbler","Cetti's Warbler","Brownish-flanked Bush Warbler","Long-tailed Tit","Bushtit","Eurasian Blackcap","Garden Warbler","Barred Warbler","Lesser Whitethroat","Western Orphean Warbler","Sardinian Warbler","Western Subalpine Warbler","Greater Whitethroat","Dartford Warbler","Wrentit","Indian White-eye","Indian Scimitar-Babbler","Puff-throated Babbler","Brown-cheeked Fulvetta","Streaked Laughingthrush","Chestnut-crowned Laughingthrush","Large Gray Babbler","Jungle Babbler","Common Babbler","Ruby-crowned Kinglet","Golden-crowned Kinglet","Goldcrest","Common Firecrest","Eurasian Nuthatch","Red-breasted Nuthatch","White-breasted Nuthatch","Pygmy Nuthatch","Brown-headed Nuthatch","Eurasian Treecreeper","Brown Creeper","Short-toed Treecreeper","Long-billed Gnatwren","Masked Gnatcatcher","Tropical Gnatcatcher","Blue-gray Gnatcatcher","Black-tailed Gnatcatcher","California Gnatcatcher","Black-capped Gnatcatcher","Rock Wren","Canyon Wren","House Wren","Eurasian Wren","Pacific Wren","Winter Wren","Sedge Wren","Grass Wren","Marsh Wren","Carolina Wren","Bewick's Wren","Rufous-naped Wren","Cactus Wren","Thrush-like Wren","Spot-breasted Wren","Happy Wren","Sinaloa Wren","Banded Wren","Rufous-and-white Wren","Cabanis's Wren","White-bellied Wren","White-breasted Wood-Wren","Gray-breasted Wood-Wren","American Dipper","European Starling","Spotless Starling","Common Myna","Gray Catbird","Curve-billed Thrasher","Brown Thrasher","Long-billed Thrasher","Bendire's Thrasher","California Thrasher","LeConte's Thrasher","Crissal Thrasher","Sage Thrasher","Chalk-browed Mockingbird","Tropical Mockingbird","Northern Mockingbird","Eastern Bluebird","Western Bluebird","Mountain Bluebird","Townsend's Solitaire","Brown-backed Solitaire","Andean Solitaire","Varied Thrush","Orange-billed Nightingale-Thrush","Veery","Gray-cheeked Thrush","Bicknell's Thrush","Swainson's Thrush","Hermit Thrush","Wood Thrush","Mistle Thrush","Song Thrush","Redwing","Eurasian Blackbird","Pale-breasted Thrush","White-throated Thrush","Rufous-bellied Thrush","Clay-colored Thrush","American Robin","Rufous-backed Robin","Austral Thrush","Creamy-bellied Thrush","Chiguanco Thrush","Fieldfare","Ring Ouzel","Asian Brown Flycatcher","Spotted Flycatcher","Rufous-tailed Scrub-Robin","Oriental Magpie-Robin","European Robin","Thrush Nightingale","Common Nightingale","Bluethroat","Malabar Whistling-Thrush","Blue Whistling-Thrush","Taiga Flycatcher","Red-breasted Flycatcher","European Pied Flycatcher","Collared Flycatcher","Common Redstart","Black Redstart","Blue Rock-Thrush","Whinchat","European Stonechat","Pied Bushchat","Gray Bushchat","Northern Wheatear","Brown Rock Chat","Bohemian Waxwing","Cedar Waxwing","Phainopepla","Pale-billed Flowerpecker","Purple-rumped Sunbird","Crimson-backed Sunbird","Purple Sunbird","Loten's Sunbird","Olive Warbler","Scaly-breasted Munia","Common Waxbill","Red Avadavat","Dunnock","House Sparrow","Spanish Sparrow","Eurasian Tree Sparrow","Rock Sparrow","Gray Wagtail","Western Yellow Wagtail","Citrine Wagtail","White-browed Wagtail","White Wagtail","Paddyfield Pipit","Tawny Pipit","Meadow Pipit","Tree Pipit","Red-throated Pipit","Water Pipit","American Pipit","Sprague's Pipit","Common Chaffinch","Brambling","Elegant Euphonia","Purple-throated Euphonia","Yellow-throated Euphonia","Evening Grosbeak","Hawfinch","Common Rosefinch","Pine Grosbeak","Eurasian Bullfinch","Gray-crowned Rosy-Finch","House Finch","Purple Finch","Cassin's Finch","European Greenfinch","Eurasian Linnet","Common Redpoll","Lesser Redpoll","Red Crossbill","Cassia Crossbill","White-winged Crossbill","European Goldfinch","European Serin","Island Canary","Eurasian Siskin","Pine Siskin","Lesser Goldfinch","Lawrence's Goldfinch","American Goldfinch","Hooded Siskin","Lapland Longspur","Chestnut-collared Longspur","Thick-billed Longspur","Snow Bunting","Corn Bunting","Rock Bunting","Cirl Bunting","Yellowhammer","Ortolan Bunting","Reed Bunting","Common Chlorospingus","Rufous-winged Sparrow","Botteri's Sparrow","Cassin's Sparrow","Bachman's Sparrow","Grasshopper Sparrow","Grassland Sparrow","Yellow-browed Sparrow","Olive Sparrow","Green-backed Sparrow","Chipping Sparrow","Clay-colored Sparrow","Black-chinned Sparrow","Field Sparrow","Brewer's Sparrow","Black-throated Sparrow","Lark Sparrow","Lark Bunting","Orange-billed Sparrow","Chestnut-capped Brushfinch","American Tree Sparrow","Fox Sparrow","Dark-eyed Junco","Yellow-eyed Junco","Rufous-collared Sparrow","White-crowned Sparrow","Golden-crowned Sparrow","Harris's Sparrow","White-throated Sparrow","Sagebrush Sparrow","Bell's Sparrow","Vesper Sparrow","LeConte's Sparrow","Seaside Sparrow","Nelson's Sparrow","Savannah Sparrow","Baird's Sparrow","Henslow's Sparrow","Song Sparrow","Lincoln's Sparrow","Swamp Sparrow","Canyon Towhee","Abert's Towhee","California Towhee","Rufous-crowned Sparrow","Green-tailed Towhee","Spotted Towhee","Eastern Towhee","Yellow-breasted Chat","Yellow-headed Blackbird","Bobolink","Western Meadowlark","Eastern Meadowlark","Long-tailed Meadowlark","Yellow-winged Cacique","Russet-backed Oropendola","Crested Oropendola","Montezuma Oropendola","Golden-winged Cacique","Orchard Oriole","Hooded Oriole","Streak-backed Oriole","Bullock's Oriole","Altamira Oriole","Audubon's Oriole","Baltimore Oriole","Scott's Oriole","Red-winged Blackbird","Brown-headed Cowbird","Melodious Blackbird","Rusty Blackbird","Brewer's Blackbird","Common Grackle","Boat-tailed Grackle","Great-tailed Grackle","Austral Blackbird","Chopi Blackbird","Ovenbird","Worm-eating Warbler","Louisiana Waterthrush","Northern Waterthrush","Golden-winged Warbler","Blue-winged Warbler","Black-and-white Warbler","Prothonotary Warbler","Swainson's Warbler","Crescent-chested Warbler","Tennessee Warbler","Orange-crowned Warbler","Lucy's Warbler","Nashville Warbler","Virginia's Warbler","Connecticut Warbler","Gray-crowned Yellowthroat","MacGillivray's Warbler","Mourning Warbler","Kentucky Warbler","Common Yellowthroat","Hooded Warbler","American Redstart","Kirtland's Warbler","Cape May Warbler","Cerulean Warbler","Northern Parula","Tropical Parula","Magnolia Warbler","Bay-breasted Warbler","Blackburnian Warbler","Yellow Warbler","Chestnut-sided Warbler","Blackpoll Warbler","Black-throated Blue Warbler","Palm Warbler","Pine Warbler","Yellow-rumped Warbler","Yellow-throated Warbler","Prairie Warbler","Grace's Warbler","Black-throated Gray Warbler","Townsend's Warbler","Hermit Warbler","Golden-cheeked Warbler","Black-throated Green Warbler","Rufous-capped Warbler","Chestnut-capped Warbler","Golden-crowned Warbler","Three-striped Warbler","Flavescent Warbler","White-browed Warbler","Buff-rumped Warbler","Russet-crowned Warbler","Canada Warbler","Wilson's Warbler","Red-faced Warbler","Painted Redstart","Slate-throated Redstart","Hepatic Tanager","Summer Tanager","Scarlet Tanager","Western Tanager","Red-crowned Ant-Tanager","Red-throated Ant-Tanager","Northern Cardinal","Pyrrhuloxia","Rose-breasted Grosbeak","Black-headed Grosbeak","Blue Grosbeak","Lazuli Bunting","Indigo Bunting","Varied Bunting","Painted Bunting","Dickcissel","Red-crested Cardinal","Scarlet-rumped Tanager","Silver-beaked Tanager","Blue-gray Tanager","Sayaca Tanager","Palm Tanager","Masked Flowerpiercer","Diuca Finch","Mourning Sierra Finch","Black-and-rufous Warbling Finch","Saffron Finch","Wedge-tailed Grass-Finch","Blue-black Grassquit","Morelet's Seedeater","Yellow-bellied Seedeater","Bananaquit","Yellow-faced Grassquit","Buff-throated Saltator","Black-headed Saltator","Cinnamon-bellied Saltator","Blue-gray Saltator","Green-winged Saltator"]
geoMapping = [None,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,
              21,22,23,24,25,26,27,28,29,30,31,32,33,35,36,37,38,39,40,
              41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,
              61,62,63,64,65,66,67,69,70,71,72,73,74,75,76,77,78,79,80,
              81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,
              101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,
              121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,
              141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,
              161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,
              181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199,200,
              201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,
              221,222,223,224,225,226,227,228,229,230,231,232,233,234,235,236,237,238,239,240,
              241,242,243,244,245,246,247,248,249,250,251,252,253,254,255,256,257,258,259,260,
              261,262,263,264,265,266,267,268,269,270,271,272,273,274,275,276,277,278,279,280,
              281,282,283,284,285,286,287,288,289,290,291,292,293,294,295,296,297,298,299,300,
              301,302,303,304,305,306,307,308,309,310,311,312,313,314,315,316,317,318,319,320,
              321,322,323,324,325,326,327,328,329,330,331,333,334,335,336,337,338,339,340,
              341,342,343,344,345,346,347,348,349,350,351,352,353,354,355,356,357,358,359,360,
              361,362,363,364,365,366,367,368,369,370,371,372,373,374,375,376,377,378,379,380,
              381,382,383,384,385,386,387,388,389,390,391,392,393,394,395,396,397,398,399,400,
              401,402,403,404,405,406,407,408,409,410,411,412,413,414,415,416,417,418,419,420,
              421,422,423,424,425,426,427,428,429,430,431,432,433,434,435,436,437,438,439,440,
              441,442,443,444,445,446,447,448,449,450,451,452,453,454,455,456,457,458,459,460,
              461,462,463,464,465,466,467,468,469,470,471,472,473,474,475,476,477,478,479,480,
              482,483,484,485,486,487,488,489,490,492,493,494,495,496,497,498,499,500,
              501,502,503,504,505,506,507,508,509,510,511,512,513,514,515,516,517,518,519,520,
              521,522,523,524,525,526,527,528,529,530,531,532,533,534,535,536,537,538,539,540,
              541,542,543,544,545,546,547,548,549,550,551,552,553,554,555,556,557,558,559,560,
              561,562,563,564,565,566,567,568,569,572,573,574,575,576,577,578,579,580,
              581,582,583,584,585,586,587,588,589,590,591,592,593,594,595,596,597,598,599,600,
              601,602,604,605,606,607,608,609,611,612,613,614,615,616,617,618,619,620,
              621,622,623,624,625,626,627,628,629,630,631,632,633,634,635,636,637,638,639,640,
              641,643,644,645,646,647,649,651,652,653,654,655,656,657,658,659,660,
              661,662,663,664,665,666,667,668,669,670,671,672,673,674,675,676,677,678,679,680,
              681,682,683,684,685,686,687,688,689,690,691,692,693,694,695,696,697,698,699,700,
              701,702,703,704,705,706,707,708,709,710,711,712,713,714,715,716,717,718,719,720,
              721,722,723,724,725,726,727,728,729,730,731,732,733,734,735,736,737,738,739,740,
              741,742,743,744,745,746,747,748,749,750,751,752,753,754,755,756,757,759,760,
              761,762,763,764,765,766,767,768,769,770,771,772,773,774,775,776,777,778,779,780,
              781,782,783,784,785,786,787,788,789,790,791,792,793,794,795,796,797,798,799,800,
              801,802,803,804,805,806,807,808,809,810,811,812,813,814,815,816,817,818,819,820,
              821,822,823,824,825,826,827,828,829,830,831,832,833,834,835,836,837,838,839,840,
              841,842,843,844,845,846,847,848,849,850,851,852,853,854,855,856,857,858,859,860,
              861,862,863,864,865,866,867,868,869,870,871,872,873,874,875,876,877,878,879,880,
              881,882,883,884,885,886,887,888,889,890,891,892,893,894,895,896,897,898,899,900,
              901,902,903,904,905,906,907,908,909,910,911,912,913,914,915,916,917,918,919,920,
              921,922,923,924,925,926,927,928,929,930,931,932,933,934,935,936,937,938,939,940,
              941,942,943,944,945,946,947,948,949,950,951,952,953,954,955,956,957,958,959,960,
              961,962,963,964,965,966,967,968,969,971,972,973,974,975,976,977,978,979,980,
              981,982,983,984,985,986,987,989,990,991,992,993,994,995,996,997,998,999,1000,
              1001,1002,1003,1004,1005,1006,1007,1008,1009,1010,1011,1012,1013,1014,1015,1016,1017,1018,1019,1020,
              1021,1022,1023,1024,1025,1026,1027,1028,1030,1031,1032,1033,1034,1035,1036,1037,1038,1039,1040,
              1041,1042,1043,1044,1045,1046,1047,1048,1049,1050,1051,1052,1053,1054,1055,1056,1057,1058,1059,1060,
              1061,1062,1063,1064,1066,1067,1068,1069,1070]  # 332, 481, 491, 570, 571, 603, 610, 642, 648, 650, 758, 970, 988, 1029?

userDir = os.path.expanduser('~')
PREDICTED_SPECIES = {}

global SPECTROGRAM_INTERPRETER, SPEC_INPUT_LAYER, SPEC_OUTPUT_LAYER,\
       SOUND_INTERPRETER, SOUND_INPUT_LAYER, SOUND_OUTPUT_LAYER, \
       M_INTERPRETER, M_INPUT_LAYER, M_OUTPUT_LAYER


def read_audio_data(path, load_type, sample_rate=22050):
    start_total = time.time()
    print('READING AUDIO DATA...', end=' ', flush=True)

    start_read = time.time()
    # Open file with librosa (uses ffmpeg or libav)
    samples, _ = librosa.load(path, sr=sample_rate, res_type=load_type)
    print('DONE! Time', int((time.time() - start_read) * 10) / 10.0, 'SECONDS')

    # Calculate number of 3-second chunks based on recording duration
    duration = librosa.get_duration(y=samples, sr=sample_rate)
    chunks = math.ceil(np.round(duration, 0)/3)

    print('DONE! READ', str(chunks), 'CHUNKS.')
    print('DONE! Time', int((time.time() - start_total) * 10) / 10.0, 'SECONDS')
    return chunks, samples


def load_model(tf_model):
    print(f'LOADING TF LITE {tf_model} MODEL...', end=' ')

    # Load TFLite model and allocate tensors.
    url = '/model/' + tf_model + '.tflite'
    model_path = open(url)
    myinterpreter = tf.lite.Interpreter(model_path=model_path)
    myinterpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = myinterpreter.get_input_details()
    output_details = myinterpreter.get_output_details()

    # Get input tensor index
    input_layer_index = input_details[0]['index']
    output_layer_index = output_details[0]['index']

    print('DONE!')
    return myinterpreter, input_layer_index, output_layer_index


def predict_filter(lat, lon, week):

    # Get input tensor
    input_details = M_INTERPRETER.get_input_details()

    # Run inference
    M_INTERPRETER.set_tensor(input_details[0]['index'], [np.float32(lon)])
    M_INTERPRETER.set_tensor(input_details[1]['index'], [np.float32(week)])
    M_INTERPRETER.set_tensor(input_details[2]['index'], [np.float32(lat)])
    M_INTERPRETER.invoke()

    return M_INTERPRETER.get_tensor(M_OUTPUT_LAYER)[0]  # geo_output


def predict_species_list(lat, lon, week, sf_thresh):

    global PREDICTED_SPECIES

    # Make filter prediction
    l_filter = predict_filter(lat, lon, week)

    meta_data = [l_filter[i] if i is not None else None for i in geoMapping]

    # Zip with labels
    meta_data = dict(zip(classifications, meta_data))

    # Apply threshold
    slist = {k: v for (k, v) in meta_data.items() if v is not None and v >= float(sf_thresh)}

    # Sort by filter value
    PREDICTED_SPECIES = dict(sorted(slist.items(), key=lambda x: x[0]))


def run_model(sample, interpreter, input_layer, output_layer):

    # Make prediction
    interpreter.set_tensor(input_layer, sample)
    interpreter.invoke()  # Run inference!
    output = interpreter.get_tensor(output_layer)  # output, spectrogram or sound_id interpreter

    return output


def create_image(sig, sample_index, image, spectrogram_interpreter, spectrogram_input, spectrogram_output):

    # Split sample by sampleHop and interpret duration of input
    sample_hop = 128
    spectrogram_model_input_size = 512
    for i in range(0, spectrogram_model_input_size):
        split = sig[sample_index:sample_index + spectrogram_model_input_size]
        sample_index += sample_hop
        output_data = run_model(split, spectrogram_interpreter, spectrogram_input,
                                spectrogram_output).flatten()
        image[:, i] = output_data

    return image, sample_index


def analyze_audio_data(chunks, sig, lat, lon, week, min_conf, sf_thresh):

    detections = pd.DataFrame(columns=['Time Vocalized', 'Species', 'Confidence', 'Species Frequency'])
    start = time.time()
    print(f'ANALYZING AUDIO...', end=' ', flush=True)

    if len(PREDICTED_SPECIES) == 0:
        predict_species_list(lat, lon, week, sf_thresh)
    # print(PREDICTED_SPECIES)
    rcnt = 0

    # Spectrogram generation vars
    image_height = 128
    image_width = 512
    sample_index = 0

    # Parse every chunk
    for c in range(1, chunks):

        # Run inference on a 3-second clip
        image = np.zeros((image_height, image_width), dtype=np.float32)

        image, sample_index = create_image(sig, sample_index, image, SPECTROGRAM_INTERPRETER,
                                           SPEC_INPUT_LAYER, SPEC_OUTPUT_LAYER)

        color_image = np.stack((image, image, image), axis=-1)
        normalized_image = color_image / 255.0
        reshaped_image = normalized_image.reshape(1, image_height, image_width, 3)

        prediction = run_model(reshaped_image, SOUND_INTERPRETER, SOUND_INPUT_LAYER, SOUND_OUTPUT_LAYER)[0]

        sample_hop = 128
        rate = 22050
        for (p, score) in enumerate(list(prediction)):
            if score >= min_conf and ((classifications[p] in PREDICTED_SPECIES or len(PREDICTED_SPECIES) == 0)
                                      and (classifications[p] != 'bird sp.')):
                frequency = np.round(PREDICTED_SPECIES.get(classifications[p]), 4)  # occurrence frequency species
                time_vocal = str(datetime.timedelta(seconds=(sample_index - image_width * sample_hop) / rate))
                write_results_to_file(detections, time_vocal, classifications[p],  score, frequency)
                rcnt += 1

    print('DONE! Time', int((time.time() - start) * 10) / 10.0, 'SECONDS')
    print('WROTE', rcnt, 'RESULTS.')
    return detections


def write_results_to_file(df, time_vocal, species, conf, freq):
    df.loc[len(df)] = [
        str(time_vocal).split(".")[0],
        f"{species}",
        f"{(conf*100):.1f}%",
        f"{(freq*100):.1f}%"
    ]
    return df


def main():
    start = time.time()

    # Load model
    global SPECTROGRAM_INTERPRETER, SPEC_INPUT_LAYER, SPEC_OUTPUT_LAYER, \
        SOUND_INTERPRETER, SOUND_INPUT_LAYER, SOUND_OUTPUT_LAYER, \
        M_INTERPRETER, M_INPUT_LAYER, M_OUTPUT_LAYER

    SPECTROGRAM_INTERPRETER, SPEC_INPUT_LAYER, SPEC_OUTPUT_LAYER = load_model(tf_model='msid685v4_spectrogram')
    SOUND_INTERPRETER, SOUND_INPUT_LAYER, SOUND_OUTPUT_LAYER = load_model(tf_model='sound_id_v31')
    M_INTERPRETER, M_INPUT_LAYER, M_OUTPUT_LAYER = load_model(tf_model='geo_v31')


    st.sidebar.title("Data Selections")

    date = st.sidebar.date_input("Date of recording (MM/DD/YYYY)", value=datetime.datetime.now(), format="MM/DD/YYYY", key="date")
    week_of_year = date.isocalendar()[1]

    latitude = st.sidebar.number_input("Latitude", value=37.299, key="lat")
    longitude = st.sidebar.number_input("Longitude", value=-75.929, key="lon")

    min_conf = st.sidebar.number_input("Minimum Confidence Percentage", value=30,
                                       min_value=0, max_value=100, key="min_conf") / 100

    st.sidebar.number_input("Minimum Species Frequency", value=3,
                                        min_value=0, max_value=100, key="sf_thresh")
    sf_thresh = st.session_state.sf_thresh / 100

    load = st.sidebar.selectbox("Resample Type", ["soxr_vhq", "soxr_hq", "kaiser_best", "kaiser_fast"], key="res_type", index=1)

    run = st.sidebar.button("Run", key="run")


    custom_holder = st.empty()
    with custom_holder.container():
        st.title("Upload WAV file")
        uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None and run:
        custom_holder.empty()

        st.title("Results")

        # Read Audio
        chunks, audio_data = read_audio_data(uploaded_file, load)

        # Analyze audio data and get detections
        st.dataframe(analyze_audio_data(chunks, audio_data, latitude, longitude, week_of_year, min_conf, sf_thresh),
                     hide_index=True)

        print('DONE! Total time', int((time.time() - start) * 10) / 10.0, 'SECONDS')

        if st.button("Reset", type="primary"):
            st.rerun()


# Run main
if __name__ == "__main__":  # use for testing
    st.set_page_config(page_icon='🐧', initial_sidebar_state='expanded')
    st.markdown("""<style>
                body {text-align: center}
                p {text-align: center} 
                button {float: center} 
                [data-testid=stSidebarUserContent] {
                    margin-top: -75px;
                }
                </style>""", unsafe_allow_html=True)
    main()
