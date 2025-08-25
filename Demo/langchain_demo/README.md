# ospo-ai-agent

### LangChain Demo 

The LangChain demo is in this directory. Check out the [langchain_pace_demo_m2.ipynb](https://github.com/gt-ospo/ospo-ai-agent/blob/main/Demo/langchain_demo/langchain_pace_demo_m2.ipynb).

### SMARTech Scraper Notes

The file essentially scrapes in batches of 20 from each "page" listing (i.e 1-20, 21-40, etc) and downloads all pdfs associated with each item to a local dir. You can adjust the max number of pages. The scraper requests: item_id (id of the paper) -> bundle_id (multiple per item) -> bitsream_id (multiple per bundle) from the servers REST endpoints. SMARTech uses the DSpace 7 open‑source digital asset management system backend (which is what the retrieval flow is based on). 

For context a bundle looks like:
```json
"bundles": [
      {
        "uuid": "141a4844-3453-4710-ae1d-0c03dcef7cef",
        "name": "ORIGINAL",
        "type": "bundle",
        "_links": {
          "bitstreams": {
            "href": "https://repository.gatech.edu/server/api/core/bundles/141a4844-3453-4710-ae1d-0c03dcef7cef/bitstreams"
          }
        }
      },
```
Each item has multiple bundles like: ORIGINAL, THUMBNAIL, LICENSE. The pdf we want is usually in ORIGINAL (but we grab all files in all bundles anyway just to be safe). Then each bundle has multiple bistreams, where a "bitstream" represents an actual file in a bundle. The scraper then appends /content to the href for that file and downloads it with a GET request at: 
```python
url = f"{BASE}/server/api/core/bitstreams/{bs['uuid']}/content"
```
