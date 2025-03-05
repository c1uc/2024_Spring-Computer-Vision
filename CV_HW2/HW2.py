import cv2
import numpy as np
import random
import os


# create a window to show the image
# It will show all the windows after you call im_show()
# Remember to call im_show() in the end of main
def creat_im_window(window_name, img):
    cv2.imshow(window_name, img)


# show the all window you call before im_show()
# and press any key to close all windows
def im_show():
    cv2.waitKey(0)
    cv2.destroyAllWindows()


class RANSAC:
    def __init__(self, threshold=0.75, iter_num=10000, ransac_tolerance=3, image_list=None, early_termination_ratio=0.9):
        self.threshold = threshold
        self.iter_num = iter_num
        self.ransac_tolerance = ransac_tolerance
        self.h = None
        self.image_list = image_list
        self.sift = cv2.SIFT_create()
        self.early_termination_ratio = early_termination_ratio

    def find_knn(self, img1, img2):
        kp1, des1 = self.sift.detectAndCompute(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), None)
        kp2, des2 = self.sift.detectAndCompute(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), None)
        pts1, pts2 = [], []

        for kp, des in zip(kp1, des1):
            matches = [(kp_, np.linalg.norm(des - des_)) for kp_, des_ in zip(kp2, des2)]
            matches.sort(key=lambda x: x[1])
            if matches[0][1] < self.threshold * matches[1][1]:
                pts1.append(kp.pt)
                pts2.append(matches[0][0].pt)

        return pts1, pts2

    def ransac(self, pts1, pts2):
        max_in = 0
        best_h = None

        n = len(pts1)
        for _ in range(self.iter_num):
            idx = random.sample(range(n), 4)
            pts1_ = np.array([pts1[i] for i in idx])
            pts2_ = np.array([pts2[i] for i in idx])
            h = self.calc_h(pts1_, pts2_)

            pts1__ = np.append(pts1, np.ones((n, 1)), axis=1)

            pts_h = h @ pts1__.T
            pts_h = pts_h / pts_h[2]
            pts_h = pts_h[:2].T

            in_count = np.sum(np.linalg.norm(pts_h - pts2, axis=1) < self.ransac_tolerance)

            if in_count > max_in:
                max_in = in_count
                best_h = h

            if max_in / n > self.early_termination_ratio:
                break

        self.h = best_h

    @staticmethod
    def calc_h(pts1, pts2):
        a = np.zeros((8, 8))
        b = np.zeros(8)

        a[:4, :2] = pts1[:4]
        a[:4, 2] = 1
        a[4:, 3:5] = pts1[:4]
        a[4:, 5] = 1
        a[:4, 6] = -pts1[:4, 0] * pts2[:4, 0]
        a[:4, 7] = -pts1[:4, 1] * pts2[:4, 0]
        a[4:, 6] = -pts1[:4, 0] * pts2[:4, 1]
        a[4:, 7] = -pts1[:4, 1] * pts2[:4, 1]

        b[:4] = pts2[:4, 0]
        b[4:] = pts2[:4, 1]

        h = np.linalg.lstsq(a, b, rcond=None)[0]
        h = np.append(h, 1).reshape(3, 3)
        return h

    @staticmethod
    def calc_mask(img):
        mask = img != 0
        direction = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]])
        for d in direction:
            mask &= np.roll(img, d, axis=(0, 1)) != 0
        return mask

    @staticmethod
    def no_blending(img_left, img_right, mask_left, mask_right, mask):
        img_result = img_right.copy()
        img_result[mask_left] = img_left[mask_left]
        img_result[mask_left & mask_right] = img_left[mask_left & mask_right]

        return img_result

    @staticmethod
    def linear_blending(img_left, img_right, mask_left, mask_right, mask):
        leftmost = np.min(np.where(mask)[1])
        rightmost = np.max(np.where(mask)[1])

        step = 1 / (rightmost - leftmost)
        alpha_mask = np.zeros_like(mask, dtype=float)

        img_result = img_right.copy()
        img_result[mask_left] = img_left[mask_left]

        for i in range(leftmost, rightmost):
            alpha_mask[:, i] = (i - leftmost) * step

        alpha_mask_3d = alpha_mask[..., np.newaxis]
        one_minus_alpha_mask_3d = 1 - alpha_mask_3d

        img_result[mask] = img_left[mask] * one_minus_alpha_mask_3d[mask] + img_right[mask] * alpha_mask_3d[mask]

        return img_result

    @staticmethod
    def weighted_blending(img_left, img_right, mask_left, mask_right, mask):
        img_result = img_right.copy()
        img_result[mask_left] = img_left[mask_left]
        img_result[mask_left & mask_right] = cv2.addWeighted(img_left, 0.5, img_right, 0.5, 0)[mask_left & mask_right]

        return img_result

    def stitch(self, base_img, addition_img, base_side="right"):
        gray_base = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
        gray_addition = cv2.cvtColor(addition_img, cv2.COLOR_BGR2GRAY)

        mask_base = self.calc_mask(gray_base)
        mask_addition = self.calc_mask(gray_addition)
        mask_overlap = mask_base & mask_addition

        if base_side == "right":
            img_result = self.linear_blending(addition_img, base_img, mask_addition, mask_base, mask_overlap)
        else:
            img_result = self.linear_blending(base_img, addition_img, mask_base, mask_addition, mask_overlap)

        return img_result

    def sift_and_stitch(self, base_img, addition_img, base_side="right"):
        h1, w1 = base_img.shape[:2]
        h2, w2 = addition_img.shape[:2]
        pts_base, pts_addition = self.find_knn(base_img, addition_img)

        self.ransac(pts_addition, pts_base)

        base_corners = np.array([[0, 0], [0, h1], [w1, h1], [w1, 0]], dtype=float)
        addition_corners = np.array([[0, 0], [0, h2], [w2, h2], [w2, 0]], dtype=float)
        addition_corners = cv2.perspectiveTransform(addition_corners.reshape(1, -1, 2), self.h).reshape(-1, 2)

        min_x, min_y = np.min(np.vstack([base_corners, addition_corners]), axis=0)
        max_x, max_y = np.max(np.vstack([base_corners, addition_corners]), axis=0)
        new_size = (int(max_x - min_x), int(max_y - min_y))

        affine = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]], dtype=float)

        img_after = cv2.warpPerspective(addition_img, affine @ self.h, new_size)
        base_img_after = cv2.warpPerspective(base_img, affine, new_size)

        stitched = self.stitch(base_img_after, img_after, base_side=base_side)

        return stitched

    def __call__(self):
        mid = len(self.image_list) // 2
        base = self.image_list[mid]
        for image in self.image_list[mid - 1::-1]:
            base = self.sift_and_stitch(base, image, "right")

        for image in self.image_list[mid + 1:]:
            base = self.sift_and_stitch(base, image, "left")

        return base


if __name__ == '__main__':
    task = "Base"  # Base Challenge
    folder_path = f"./Photos/{task}/"
    img_names = os.listdir(folder_path)
    images = [cv2.imread(folder_path + img_name) for img_name in img_names]

    ransac = RANSAC(image_list=images)
    result = ransac()

    cv2.imwrite(f"result_{task}.jpg", result)
