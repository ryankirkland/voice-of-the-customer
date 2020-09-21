from bs4 import BeautifulSoup
import pandas as pd
import requests
import time
import sys


def reviews_scraper(asin_list, filename):
    '''
    Takes a list of asins, retrieves html for reviews page, and parses out key data points
    Parameters
    ----------
    List of ASINs (list of strings)
    Returns:
    -------
    review information (list), reviews_df (Pandas DataFrame)
    '''
    asin_list = [asin_list]
    print(asin_list)
    reviews = []

    headers = {"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:66.0) Gecko/20100101 Firefox/66.0", "Accept-Encoding":"gzip, deflate", "Accept":"text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8", "DNT":"1","Connection":"close", "Upgrade-Insecure-Requests":"1"}

    for asin in asin_list:
        print(f'Collecting reviews for {asin}')
        passed_last_page = None
        counter = 1
        while (passed_last_page == None) and (counter <= 10):
            print(len(reviews))

            reviews_url = f'https://www.amazon.com/product-reviews/{asin}/ref=cm_cr_arp_d_viewopt_srt?ie=UTF8&reviewerType=all_reviews&sortBy=recent&pageNumber={counter}'
            print(reviews_url)
            rev = requests.get(reviews_url, headers=headers)
            print(rev.status_code)
            reviews_page_content = rev.content

            review_soup = BeautifulSoup(reviews_page_content, features='lxml')
            print(review_soup)
            passed_last_page = review_soup.find('div', attrs={'class': 'a-section a-spacing-top-large a-text-center no-reviews-section'})

            if passed_last_page == None:

                for d in review_soup.findAll('div', attrs={'data-hook':'review'}):
                #         print(d)
                        try:
                            date = d.find('span', attrs={'data-hook':'review-date'})
                            date = date.text.split(' ')[-3:]
                            date = ' '.join(date)
                        except:
                            date = 'null'
                        try:
                            title = d.find('a', attrs={'data-hook': 'review-title'})
                        except:
                            title = 'null'
                        try:
                            product = d.find('a', attrs={'data-hook': 'format-strip'})
                            product = product.text
                        except:
                            product = 'null'
                        try:
                            review_asin = product['href'].split('/')[3]
                        except:
                            review_asin = asin
                        try:
                            verified = d.find('span', attrs={'data-hook':'avp-badge'})
                            if verified == None:
                                verified = 'Not Verified'
                            else:
                                verified = verified.text
                        except:
                            verified = 'null'
                        try:
                            description = d.find('span', attrs={'data-hook': 'review-body'})
                        except:
                            description = 'null'
                        try:
                            reviewer_name = d.find('span', attrs={'class': 'a-profile-name'})
                        except:
                            reviewer_name = 'null'
                        try:
                            stars = d.find('span', attrs={'class': 'a-icon-alt'})
                        except:
                            stars = 'null'
                        reviews.append([review_asin, product, date, verified, title.text, description.text, reviewer_name.text, float(stars.text[0:3])])
            else:
                pass

            counter += 1

            time.sleep(15)
        
    reviews_df = pd.DataFrame(reviews, columns=['asin','product','date', 'verified', 'title', 'desc', 'reviewer_name', 'rating'])

    reviews_df.to_csv(f'data/reviews/{filename}')

    print(f'{len(reviews)} reviews for {len(asin_list)} asins stored successfully in {filename}')

    return reviews, reviews_df

if __name__ == '__main__':
    reviews_scraper(*sys.argv[1:])