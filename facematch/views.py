from rest_framework.views import APIView
from rest_framework.response import Response
from django.shortcuts import render
from .face_match import compare2face, compare2multiple
import os
from facematching.settings import BASE_DIR


class OneToOne(APIView):
    def get(self, request):
        return render(request, 'one_to_one.html')

    def post(self, request):
        img1 = request.data.get('img1')
        img2 = request.data.get('img2')
        result = {'success': False, 'message': 'Something went wrong'}
        try:
            if img1 and img2:
                img1_path = handle_uploaded_file(img1)
                img2_path = handle_uploaded_file(img2)
                similarity, is_match = compare2face(img1_path, img2_path)
                result = {'success': True, 'image1': img1.name, 'image2': img2.name, 'similarity': similarity,
                          'is_match': is_match}
                os.remove(img1_path)
                os.remove(img2_path)
            else:
                result = {'success': False, 'message': 'Please select both image'}
        except Exception as e:
            print(e)
        return Response(result)


class OneToMany(APIView):
    def get(self, request):
        return render(request, 'one_to_many.html')

    def post(self, request):
        img = request.data.get('img')
        result = {'success': False}
        try:
            if img:
                img_path = handle_uploaded_file(img)
                res_dict = compare2multiple(img_path)
                result = {'success': True, 'image': img.name, 'compare_results': res_dict}
                os.remove(img_path)
            else:
                result = {'success': False, 'message': 'Please select an image'}
        except Exception as e:
            print(e)
        return Response(result)


def handle_uploaded_file(f):
    temp_file_path = os.path.join(BASE_DIR, 'temp', f.name)
    with open(temp_file_path, 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)
    return temp_file_path
