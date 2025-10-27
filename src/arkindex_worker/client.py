#!/usr/bin/env python3
import os, json
from pathlib import Path
import requests

class ArkindexClient:
    def __init__(self, api_url, api_token, endpoints=None):
        self.api_url = api_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({'Authorization': f'Token {api_token}', 'Accept': 'application/json'})
        self.endpoints = endpoints or {}

    def _url(self, path):
        return f"{self.api_url}{path}"

    def list_items(self, workspace_id, limit=10):
        tmpl = self.endpoints.get('list_items', '/workspaces/{workspace_id}/items?status=pending&limit={limit}')
        path = tmpl.format(workspace_id=workspace_id, limit=limit)
        r = self.session.get(self._url(path)); r.raise_for_status(); return r.json()

    def download_image(self, item, download_dir):
        download_dir = Path(download_dir); download_dir.mkdir(parents=True, exist_ok=True)
        if 'image_url' in item:
            url = item['image_url']
        else:
            tmpl = self.endpoints.get('download_image', '/items/{item_id}/content')
            url = self._url(tmpl.format(item_id=item['id']))
        r = self.session.get(url, stream=True); r.raise_for_status()
        fname = item.get('filename') or f"{item.get('id','item')}.jpg"
        out = download_dir / fname
        with open(out, 'wb') as f:
            for chunk in r.iter_content(8192): f.write(chunk)
        return str(out)

    def post_annotations(self, item_id, detections):
        payload = {'annotations': detections}
        tmpl = self.endpoints.get('post_annotations', '/items/{item_id}/annotations')
        r = self.session.post(self._url(tmpl.format(item_id=item_id)), json=payload); r.raise_for_status(); return r.json()

    def mark_done(self, item_id, status='processed'):
        tmpl = self.endpoints.get('mark_done', '/items/{item_id}/status')
        r = self.session.patch(self._url(tmpl.format(item_id=item_id)), json={'status': status}); r.raise_for_status(); return r.json()
